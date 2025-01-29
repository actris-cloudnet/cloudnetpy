"""Model module, containing the :class:`Model` class."""

import numpy as np
from numpy import ma
from scipy.interpolate import interp1d

from cloudnetpy import utils
from cloudnetpy.categorize import atmos_utils
from cloudnetpy.categorize.itu import (
    calc_gas_specific_attenuation,
    calc_liquid_specific_attenuation,
    calc_saturation_vapor_pressure,
)
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.datasource import DataSource
from cloudnetpy.exceptions import ModelDataError


class Model(DataSource):
    """Model class, child of DataSource.

    Args:
        model_file: File name of the NWP model file.
        alt_site: Altitude of the site above mean sea level (m).
        options: Dictionary containing optional parameters.

    Attributes:
        source_type (str): Model type, e.g. 'gdas1' or 'ecwmf'.
        model_heights (ndarray): 2-D array of model heights (one for each time
            step).
        mean_height (ndarray): Mean of *model_heights*.
        data_sparse (dict): Model variables in common height grid but without
            interpolation in time.
        data_dense (dict): Model variables interpolated to Cloudnet's dense
            time / height grid.

    """

    fields_dense = (
        "temperature",
        "pressure",
        "rh",
        "q",
    )
    fields_sparse = (*fields_dense, "uwind", "vwind")
    fields_atten = (
        "specific_gas_atten",
        "specific_saturated_gas_atten",
        "specific_liquid_atten",
    )

    def __init__(self, model_file: str, alt_site: float, options: dict | None = None):
        super().__init__(model_file)
        self.options = options
        self.source_type = _find_model_type(model_file)
        self.model_heights = self._get_model_heights(alt_site)
        self.mean_height = _calc_mean_height(self.model_heights)
        self.height: np.ndarray
        self.data_sparse: dict = {}
        self.data_dense: dict = {}
        self._append_grid()

    def interpolate_to_common_height(self) -> None:
        """Interpolates model variables to common height grid."""

        def _interpolate_variable(data_in: ma.MaskedArray) -> CloudnetArray:
            datai = ma.zeros((len(self.time), len(self.mean_height)))
            for ind, (alt, prof) in enumerate(
                zip(self.model_heights, data_in, strict=True),
            ):
                if prof.mask.all():
                    datai[ind, :] = ma.masked
                else:
                    fun = interp1d(alt, prof, fill_value="extrapolate")
                    datai[ind, :] = fun(self.mean_height)
            return CloudnetArray(datai, key, units)

        for key in self.fields_sparse:
            variable = self.dataset.variables[key]
            data = variable[:]
            units = variable.units
            self.data_sparse[key] = _interpolate_variable(data)

    def interpolate_to_grid(
        self,
        time_grid: np.ndarray,
        height_grid: np.ndarray,
    ) -> list:
        """Interpolates model variables to Cloudnet's dense time / height grid.

        Args:
            time_grid: The target time array (fraction hour).
            height_grid: The target height array (m).

        Returns:
            Indices fully masked profiles.

        """
        half_height = height_grid - np.diff(height_grid, prepend=0) / 2
        for key in self.fields_dense + self.fields_atten:
            array = self.data_sparse[key][:]
            valid_profiles = _find_number_of_valid_profiles(array)
            if valid_profiles < 2:
                raise ModelDataError
            self.data_dense[key] = utils.interpolate_2d_mask(
                self.time,
                self.mean_height,
                array,
                time_grid,
                half_height if "atten" in key else height_grid,
            )
        self.height = height_grid
        return utils.find_masked_profiles_indices(self.data_dense["temperature"])

    def calc_wet_bulb(self) -> None:
        """Calculates wet-bulb temperature in dense grid."""
        wet_bulb_temp = atmos_utils.calc_wet_bulb_temperature(self.data_dense)
        offset = (self.options or {}).get("temperature_offset", 0)
        wet_bulb_temp += offset
        self.append_data(wet_bulb_temp, "Tw", units="K")
        if offset:
            self.data["Tw"].temperature_correction_applied = offset

    def screen_sparse_fields(self) -> None:
        """Removes model fields that we don't want to write in the output."""
        fields_to_keep = ("temperature", "pressure", "q", "uwind", "vwind")
        self.data_sparse = {key: self.data_sparse[key] for key in fields_to_keep}

    def _append_grid(self) -> None:
        self.append_data(np.array(self.time), "model_time")
        self.append_data(self.mean_height, "model_height")

    def _get_model_heights(self, alt_site: float) -> np.ndarray:
        """Returns model heights for each time step."""
        try:
            model_heights = self.dataset.variables["height"]
        except KeyError as err:
            msg = "No 'height' variable in the model file."
            raise ModelDataError(msg) from err
        return self.to_m(model_heights) + alt_site

    def calc_attenuations(self, frequency: float):
        temperature = self.getvar("temperature")
        pressure = self.getvar("pressure")
        specific_humidity = self.getvar("q")

        self.data_sparse["specific_liquid_atten"] = calc_liquid_specific_attenuation(
            temperature, frequency
        )
        vp = atmos_utils.calc_vapor_pressure(pressure, specific_humidity)
        svp = calc_saturation_vapor_pressure(temperature)
        self.data_sparse["specific_gas_atten"] = calc_gas_specific_attenuation(
            pressure, vp, temperature, frequency
        )
        self.data_sparse["specific_saturated_gas_atten"] = (
            calc_gas_specific_attenuation(pressure, svp, temperature, frequency)
        )


def _calc_mean_height(model_heights: np.ndarray) -> np.ndarray:
    mean_height = ma.mean(model_heights, axis=0)
    return np.array(mean_height)


def _find_model_type(file_name: str) -> str:
    """Finds model type from the model filename."""
    possible_keys = ("gdas1", "icon", "ecmwf", "harmonie", "era5", "arpege")
    for key in possible_keys:
        if key in file_name:
            return key
    msg = f"Unknown model type: {file_name}"
    raise ValueError(msg)


def _find_number_of_valid_profiles(array: np.ndarray) -> int:
    n_good = 0
    for row in array:
        if not hasattr(row, "mask") or np.sum(row.mask.astype(int)) == 0:
            n_good += 1
    return n_good
