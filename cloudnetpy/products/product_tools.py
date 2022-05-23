"""General helper classes and functions for all products."""
from collections import namedtuple
from typing import Dict, Union

import netCDF4
import numpy as np
from numpy import ma

from cloudnetpy import constants, utils
from cloudnetpy.categorize import atmos
from cloudnetpy.datasource import DataSource

IceCoefficients = namedtuple("IceCoefficients", "K2liquid0 ZT T Z c")


class CategorizeBits:
    """Class holding information about category and quality bits.

    Args:
        categorize_file (str): Categorize file name.

    Attributes:
        category_bits (dict): Dictionary containing boolean fields for `droplet`,
            `falling`, `cold`, `melting`, `aerosol`, `insect`.

        quality_bits (dict): Dictionary containing boolean fields for `radar`,
            `lidar`, `clutter`, `molecular`, `attenuated` and `corrected`.

    """

    category_keys = ("droplet", "falling", "cold", "melting", "aerosol", "insect")

    quality_keys = ("radar", "lidar", "clutter", "molecular", "attenuated", "corrected")

    def __init__(self, categorize_file: str):
        self._categorize_file = categorize_file
        self.category_bits = self._read_bits("category")
        self.quality_bits = self._read_bits("quality")

    def _read_bits(self, bit_type: str) -> dict:
        """Converts bitfield into dictionary."""
        with netCDF4.Dataset(self._categorize_file) as nc:
            try:
                bitfield = nc.variables[f"{bit_type}_bits"][:]
            except KeyError as err:
                raise KeyError from err
            keys = getattr(CategorizeBits, f"{bit_type}_keys")
            bits = {key: utils.isbit(bitfield, i) for i, key in enumerate(keys)}
        return bits


class ProductClassification(CategorizeBits):
    """Base class for creating different classifications in the child classes
    of various Cloudnet products. Child of CategorizeBits class.

    Args:
        categorize_file (str): Categorize file name.

    Attributes:
        is_rain (ndarray): 1D array denoting rainy profiles.

    """

    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.is_rain = get_is_rain(categorize_file)


class IceClassification(ProductClassification):
    """Class storing the information about different ice types.
    Child of ProductClassification().
    """

    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.is_ice = self._find_ice()
        self.would_be_ice = self._find_would_be_ice()
        self.corrected_ice = self._find_corrected_ice()
        self.uncorrected_ice = self._find_uncorrected_ice()
        self.ice_above_rain = self._find_ice_above_rain()
        self.cold_above_rain = self._find_cold_above_rain()

    def _find_ice(self) -> np.ndarray:
        return (
            self.category_bits["falling"]
            & self.category_bits["cold"]
            & ~self.category_bits["melting"]
            & ~self.category_bits["insect"]
        )

    def _find_would_be_ice(self) -> np.ndarray:
        warm_falling = (
            self.category_bits["falling"]
            & ~self.category_bits["cold"]
            & ~self.category_bits["insect"]
        )
        return warm_falling | self.category_bits["melting"]

    def _find_corrected_ice(self) -> np.ndarray:
        return self.is_ice & self.quality_bits["attenuated"] & self.quality_bits["corrected"]

    def _find_uncorrected_ice(self) -> np.ndarray:
        return self.is_ice & self.quality_bits["attenuated"] & ~self.quality_bits["corrected"]

    def _find_ice_above_rain(self) -> np.ndarray:
        is_rain = utils.transpose(self.is_rain)
        return (self.is_ice * is_rain) == 1

    def _find_cold_above_rain(self) -> np.ndarray:
        is_cold = self.category_bits["cold"]
        is_rain = utils.transpose(self.is_rain)
        is_cold_rain = (is_cold * is_rain) == 1
        return is_cold_rain & ~self.category_bits["melting"]


class IceSource(DataSource):
    """Base class for different ice products."""

    def __init__(self, categorize_file: str, product: str):
        super().__init__(categorize_file)
        self.wl_band = utils.get_wl_band(float(self.getvar("radar_frequency")))
        self.temperature = get_temperature(categorize_file)
        self.product = product
        self.coefficients = self._get_coefficients()

    def append_main_variable_including_rain(self, ice_classification: IceClassification) -> None:
        """Adds the main variable (including ice above rain)."""
        data_including_rain = self._convert_z()
        data_including_rain[~ice_classification.is_ice] = ma.masked
        self.append_data(data_including_rain, f"{self.product}_inc_rain")

    def append_main_variable(self, ice_classification: IceClassification) -> None:
        """Adds the main variable (excluding rain)."""
        data = ma.copy(self.data[f"{self.product}_inc_rain"][:])
        data[ice_classification.ice_above_rain] = ma.masked
        self.append_data(data, self.product)

    def append_status(self, ice_classification: IceClassification) -> None:
        """Adds the status of retrieval."""
        data = self.data[self.product][:]
        retrieval_status = np.zeros(data.shape, dtype=int)
        is_data = ~data.mask
        retrieval_status[is_data] = 1
        retrieval_status[is_data & ice_classification.corrected_ice] = 3
        retrieval_status[is_data & ice_classification.uncorrected_ice] = 2
        retrieval_status[~is_data & ice_classification.is_ice] = 4
        retrieval_status[ice_classification.cold_above_rain] = 6
        retrieval_status[ice_classification.ice_above_rain] = 5
        retrieval_status[ice_classification.would_be_ice & (retrieval_status == 0)] = 7
        self.append_data(retrieval_status, f"{self.product}_retrieval_status")

    def _get_coefficients(self) -> IceCoefficients:
        """Returns coefficients for ice effective radius retrieval.

        References:
            Hogan et.al. 2006, https://doi.org/10.1175/JAM2340.1
        """
        if self.product == "ier":
            if self.wl_band == 0:
                return IceCoefficients(0.878, -0.000205, -0.0015, 0.0016, -1.52)
            return IceCoefficients(0.669, -0.000296, -0.00193, -0.000, -1.502)
        if self.wl_band == 0:
            return IceCoefficients(0.878, 0.000242, -0.0186, 0.0699, -1.63)
        return IceCoefficients(0.669, 0.000580, -0.00706, 0.0923, -0.992)

    def _convert_z(self, z_variable: str = "Z") -> np.ndarray:
        """Calculates temperature weighted z, i.e. ice effective radius [m]."""
        assert self.product in ("iwc", "ier")
        assert z_variable in ("Z", "Z_sensitivity")
        temperature = self.temperature if z_variable == "Z" else ma.mean(self.temperature, axis=0)
        z_scaled = self.getvar(z_variable) + self._get_z_factor()
        g_to_kg = 0.001
        m_to_mu = 1e6
        scale = g_to_kg if self.product == "iwc" else 3 / (2 * constants.RHO_ICE) * m_to_mu
        return (
            10
            ** (
                self.coefficients.ZT * z_scaled * temperature
                + self.coefficients.T * temperature
                + self.coefficients.Z * z_scaled
                + self.coefficients.c
            )
            * scale
        )

    def _get_z_factor(self) -> float:
        """Returns empirical scaling factor for radar echo."""
        return float(utils.lin2db(self.coefficients.K2liquid0 / 0.93))


def get_is_rain(filename: str) -> np.ndarray:
    rain_rate = read_nc_fields(filename, "rain_rate")
    is_rain = rain_rate != 0
    assert isinstance(is_rain, ma.MaskedArray)
    is_rain[is_rain.mask] = True
    return np.array(is_rain)


def read_nc_fields(nc_file: str, names: Union[str, list]) -> Union[ma.MaskedArray, list]:
    """Reads selected variables from a netCDF file.

    Args:
        nc_file: netCDF file name.
        names: Variables to be read, e.g. 'temperature' or ['ldr', 'lwp'].

    Returns:
        ndarray/list: Array in case of one variable passed as a string.
        List of arrays otherwise.

    """
    names = [names] if isinstance(names, str) else names
    with netCDF4.Dataset(nc_file) as nc:
        data = [nc.variables[name][:] for name in names]
    return data[0] if len(data) == 1 else data


def interpolate_model(cat_file: str, names: Union[str, list]) -> Dict[str, np.ndarray]:
    """Interpolates 2D model field into dense Cloudnet grid.

    Args:
        cat_file: Categorize file name.
        names: Model variable to be interpolated, e.g. 'temperature' or ['temperature', 'pressure'].

    Returns:
        dict: Interpolated variables.

    """

    def _interp_field(var_name: str) -> np.ndarray:
        values = read_nc_fields(
            cat_file, ["model_time", "model_height", var_name, "time", "height"]
        )
        return utils.interpolate_2d(*values)

    names = [names] if isinstance(names, str) else names
    return {name: _interp_field(name) for name in names}


def get_temperature(categorize_file: str) -> np.ndarray:
    """Returns interpolated temperatures in Celsius."""
    atmosphere = interpolate_model(categorize_file, "temperature")
    return atmos.k2c(atmosphere["temperature"])
