"""General helper classes and functions for all products."""

from dataclasses import dataclass
from typing import NamedTuple

import netCDF4
import numpy as np
from numpy import ma
from numpy.typing import NDArray

from cloudnetpy import constants, utils
from cloudnetpy.categorize import atmos_utils
from cloudnetpy.datasource import DataSource


class IceCoefficients(NamedTuple):
    """Coefficients for ice effective radius retrieval."""

    K2liquid0: float
    ZT: float
    T: float
    Z: float
    c: float


@dataclass
class CategoryBits:
    droplet: NDArray[np.bool_]
    falling: NDArray[np.bool_]
    freezing: NDArray[np.bool_]
    melting: NDArray[np.bool_]
    aerosol: NDArray[np.bool_]
    insect: NDArray[np.bool_]


@dataclass
class QualityBits:
    radar: NDArray[np.bool_]
    lidar: NDArray[np.bool_]
    clutter: NDArray[np.bool_]
    molecular: NDArray[np.bool_]
    attenuated_liquid: NDArray[np.bool_]
    corrected_liquid: NDArray[np.bool_]
    attenuated_rain: NDArray[np.bool_]
    corrected_rain: NDArray[np.bool_]
    attenuated_melting: NDArray[np.bool_]
    corrected_melting: NDArray[np.bool_]


class CategorizeBits:
    def __init__(self, categorize_file: str):
        self._categorize_file = categorize_file
        self.category_bits = self._read_category_bits()
        self.quality_bits = self._read_quality_bits()

    def _read_category_bits(self) -> CategoryBits:
        with netCDF4.Dataset(self._categorize_file) as nc:
            bits = nc.variables["category_bits"][:]
            return CategoryBits(
                droplet=utils.isbit(bits, 0),
                falling=utils.isbit(bits, 1),
                freezing=utils.isbit(bits, 2),
                melting=utils.isbit(bits, 3),
                aerosol=utils.isbit(bits, 4),
                insect=utils.isbit(bits, 5),
            )

    def _read_quality_bits(self) -> QualityBits:
        with netCDF4.Dataset(self._categorize_file) as nc:
            bits = nc.variables["quality_bits"][:]
            return QualityBits(
                radar=utils.isbit(bits, 0),
                lidar=utils.isbit(bits, 1),
                clutter=utils.isbit(bits, 2),
                molecular=utils.isbit(bits, 3),
                attenuated_liquid=utils.isbit(bits, 4),
                corrected_liquid=utils.isbit(bits, 5),
                attenuated_rain=utils.isbit(bits, 6),
                corrected_rain=utils.isbit(bits, 7),
                attenuated_melting=utils.isbit(bits, 8),
                corrected_melting=utils.isbit(bits, 9),
            )


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
        self._is_attenuated = self._find_attenuated()
        self._is_corrected = self._find_corrected()
        self.is_ice = self._find_ice()
        self.would_be_ice = self._find_would_be_ice()
        self.corrected_ice = self._find_corrected_ice()
        self.uncorrected_ice = self._find_uncorrected_ice()
        self.ice_above_rain = self._find_ice_above_rain()
        self.clear_above_rain = self._find_clear_above_rain()

    def _find_clear_above_rain(self) -> np.ndarray:
        return (
            utils.transpose(self.is_rain) * ~self.is_ice
            & self.category_bits.freezing
            & ~self.category_bits.melting
        )

    def _find_attenuated(self) -> np.ndarray:
        return (
            self.quality_bits.attenuated_liquid
            | self.quality_bits.attenuated_rain
            | self.quality_bits.attenuated_melting
        )

    def _find_corrected(self) -> np.ndarray:
        return (
            self.quality_bits.corrected_liquid
            | self.quality_bits.corrected_rain
            | self.quality_bits.corrected_melting
        )

    def _find_ice(self) -> np.ndarray:
        return (
            self.category_bits.falling
            & self.category_bits.freezing
            & ~self.category_bits.melting
            & ~self.category_bits.insect
        )

    def _find_would_be_ice(self) -> np.ndarray:
        warm_falling = (
            self.category_bits.falling
            & ~self.category_bits.freezing
            & ~self.category_bits.insect
        )
        return warm_falling | self.category_bits.melting

    def _find_corrected_ice(self) -> np.ndarray:
        return self.is_ice & self._is_attenuated & self._is_corrected

    def _find_uncorrected_ice(self) -> np.ndarray:
        uncorrected_melting = (
            self.quality_bits.attenuated_melting & ~self.quality_bits.corrected_melting
        )
        uncorrected_rain = (
            self.quality_bits.attenuated_rain & ~self.quality_bits.corrected_rain
        )
        uncorrected_liquid = (
            self.quality_bits.attenuated_liquid & ~self.quality_bits.corrected_liquid
        )
        return (
            self.is_ice
            & self._is_attenuated
            & (uncorrected_melting | uncorrected_rain | uncorrected_liquid)
        )

    def _find_ice_above_rain(self) -> np.ndarray:
        is_rain = utils.transpose(self.is_rain)
        return (self.is_ice * is_rain) == 1


class IceSource(DataSource):
    """Base class for different ice products."""

    def __init__(self, categorize_file: str, product: str):
        super().__init__(categorize_file)
        self.wl_band = utils.get_wl_band(float(self.getvar("radar_frequency")))
        self.temperature = _get_temperature(categorize_file)
        self.product = product
        self.coefficients = self._get_coefficients()

    def append_icy_data(
        self,
        ice_classification: IceClassification,
    ) -> None:
        """Adds the main variable (including ice above rain)."""
        data = self._convert_z()
        data[~ice_classification.is_ice | ice_classification.uncorrected_ice] = (
            ma.masked
        )
        self.append_data(data, f"{self.product}")

    def append_status(self, ice_classification: IceClassification) -> None:
        """Adds the status of retrieval."""
        data = self.data[self.product][:]
        retrieval_status = np.zeros(data.shape, dtype=int)
        is_data = ~data.mask
        retrieval_status[is_data] = 1
        retrieval_status[is_data & ice_classification.corrected_ice] = 3
        retrieval_status[~is_data & ice_classification.is_ice] = 4
        retrieval_status[ice_classification.uncorrected_ice] = 2
        retrieval_status[ice_classification.clear_above_rain] = 6
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
        if self.product not in ("iwc", "ier"):
            msg = f"Invalid product: {self.product}"
            raise ValueError(msg)
        if z_variable not in ("Z", "Z_sensitivity"):
            msg = f"Invalid z_variable: {z_variable}"
            raise ValueError(msg)
        temperature = (
            self.temperature if z_variable == "Z" else ma.mean(self.temperature, axis=0)
        )
        z_scaled = self.getvar(z_variable) + self._get_z_factor()
        g_to_kg = 0.001
        m_to_mu = 1e6
        scale = (
            g_to_kg if self.product == "iwc" else 3 / (2 * constants.RHO_ICE) * m_to_mu
        )
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
        k2 = np.array(self.coefficients.K2liquid0) / 0.93
        return float(utils.lin2db(k2))


def get_is_rain(filename: str) -> np.ndarray:
    # TODO: Check that this is correct
    with netCDF4.Dataset(filename) as nc:
        for name in ["rain_detected", "rainfall_rate", "rain_rate"]:
            if name in nc.variables:
                data = nc.variables[name][:]
                data = data != 0
                data[data.mask] = True
                return np.array(data)
    msg = "No rain data found."
    raise ValueError(msg)


def read_nc_field(nc_file: str, name: str) -> ma.MaskedArray:
    with netCDF4.Dataset(nc_file) as nc:
        return nc.variables[name][:]


def interpolate_model(cat_file: str, names: str | list) -> dict[str, np.ndarray]:
    """Interpolates 2D model field into dense Cloudnet grid.

    Args:
        cat_file: Categorize file name.
        names: Model variable to be interpolated, e.g. 'temperature' or ['temperature',
            'pressure'].

    Returns:
        dict: Interpolated variables.

    """

    def _interp_field(var_name: str) -> np.ndarray:
        values = _read_nc_fields(
            cat_file,
            ["model_time", "model_height", var_name, "time", "height"],
        )
        return utils.interpolate_2d(*values)

    names = [names] if isinstance(names, str) else names
    return {name: _interp_field(name) for name in names}


def _read_nc_fields(nc_file: str, names: list[str]) -> list[ma.MaskedArray]:
    with netCDF4.Dataset(nc_file) as nc:
        return [nc.variables[name][:] for name in names]


def _get_temperature(categorize_file: str) -> np.ndarray:
    """Returns interpolated temperatures in Celsius."""
    atmosphere = interpolate_model(categorize_file, "temperature")
    return atmos_utils.k2c(atmosphere["temperature"])
