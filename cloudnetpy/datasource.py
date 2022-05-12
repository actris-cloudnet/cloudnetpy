"""Datasource module, containing the :class:`DataSource class.`"""
import logging
import os
from datetime import datetime
from typing import Callable, Optional, Union

import netCDF4
import numpy as np

from cloudnetpy import CloudnetArray, RadarArray, utils


class DataSource:
    """Base class for all Cloudnet measurements and model data.

    Args:
        full_path: Calibrated instrument / model NetCDF file.
        radar: Indicates if data is from cloud radar. Default is False.

    Attributes:
        filename (str): Filename of the input file.
        dataset (netCDF4.Dataset): A netCDF4 Dataset instance.
        source (str): Global attribute `source` read from the input file.
        time (ndarray): Time array of the instrument.
        altitude (float): Altitude of instrument above mean sea level (m).
        data (dict): Dictionary containing :class:`CloudnetArray` instances.

    """

    calc_wet_bulb: Callable
    add_meta: Callable
    rebin_to_grid: Callable
    interpolate_to_grid: Callable
    interpolate_to_common_height: Callable
    filter_stripes: Callable
    calc_errors: Callable
    remove_incomplete_pixels: Callable
    filter_1st_gate_artifact: Callable
    screen_sparse_fields: Callable
    filter_speckle_noise: Callable
    correct_atten: Callable
    radar_frequency: float
    data_dense: dict
    data_sparse: dict
    type: str

    def __init__(self, full_path: str, radar: bool = False):
        self.filename = os.path.basename(full_path)
        self.dataset = netCDF4.Dataset(full_path)
        self.source = getattr(self.dataset, "source", "")
        self.time = self._init_time()
        self.altitude = self._init_altitude()
        self.height = self._init_height()
        self.data: dict = {}
        self._is_radar = radar

    def getvar(self, *args) -> np.ndarray:
        """Returns data array from the source file variables.

        Returns just the data (and no attributes) from the original variables dictionary,
        fetched from the input netCDF file.

        Args:
            *args: possible names of the variable. The first match is returned.

        Returns:
            ndarray: The actual data.

        Raises:
             RuntimeError: The variable is not found.

        """
        for arg in args:
            if arg in self.dataset.variables:
                return self.dataset.variables[arg][:]
        raise RuntimeError("Missing variable in the input file.")

    def append_data(
        self,
        variable: Union[netCDF4.Variable, np.ndarray, float, int],
        key: str,
        name: Optional[str] = None,
        units: Optional[str] = None,
    ):
        """Adds new CloudnetVariable or RadarVariable into `data` attribute.

        Args:
            variable: netCDF variable or data array to be added.
            key: Key used with *variable* when added to `data` attribute (dictionary).
            name: CloudnetArray.name attribute. Default value is *key*.
            units: CloudnetArray.units attribute.

        """
        array_type = RadarArray if self._is_radar else CloudnetArray
        self.data[key] = array_type(variable, name or key, units)

    def get_date(self) -> list:
        """Returns date components.

        Returns:
            list: Date components [YYYY, MM, DD].

        Raises:
             RuntimeError: Not found or invalid date.

        """
        try:
            year = str(self.dataset.year)
            month = str(self.dataset.month).zfill(2)
            day = str(self.dataset.day).zfill(2)
            datetime.strptime(f"{year}{month}{day}", "%Y%m%d")
        except (AttributeError, ValueError) as read_error:
            raise RuntimeError("Missing or invalid date in global attributes.") from read_error
        return [year, month, day]

    def close(self) -> None:
        """Closes the open file."""
        self.dataset.close()

    @staticmethod
    def km2m(var: netCDF4.Variable) -> np.ndarray:
        """Converts km to m."""
        alt = var[:]
        if var.units == "km":
            alt *= 1000
        return alt

    @staticmethod
    def m2km(var: netCDF4.Variable) -> np.ndarray:
        """Converts m to km."""
        alt = var[:]
        if var.units == "m":
            alt /= 1000
        return alt

    def _init_time(self) -> np.ndarray:
        time = self.getvar("time")
        if max(time) > 25:
            logging.warning("Assuming time as seconds, converting to fraction hour")
            time = utils.seconds2hours(time)
        return time

    def _init_altitude(self) -> Union[float, None]:
        """Returns altitude of the instrument (m)."""
        if "altitude" in self.dataset.variables:
            altitude_above_sea = self.km2m(self.dataset.variables["altitude"])
            return float(np.mean(altitude_above_sea))
        return None

    def _init_height(self) -> Union[np.ndarray, None]:
        """Returns height array above mean sea level (m)."""
        if "height" in self.dataset.variables:
            return self.km2m(self.dataset.variables["height"])
        if "range" in self.dataset.variables and self.altitude is not None:
            range_instrument = self.km2m(self.dataset.variables["range"])
            return np.array(range_instrument + self.altitude)
        return None

    def _variables_to_cloudnet_arrays(self, keys: tuple) -> None:
        """Transforms netCDF4-variables into CloudnetArrays.

        Args:
            keys: netCDF4-variables to be converted. The results are saved in *self.data*
                dictionary with *fields* strings as keys.

        Notes:
            The attributes of the variables are not copied. Just the data.

        """
        for key in keys:
            self.append_data(self.dataset.variables[key], key)

    def _unknown_variable_to_cloudnet_array(
        self,
        possible_names: tuple,
        key: str,
        units: Optional[str] = None,
        ignore_mask: bool = False,
    ):
        """Transforms single netCDF4 variable into CloudnetArray.

        Args:
            possible_names: Tuple of strings containing the possible names of the variable in the
                input NetCDF file.
            key: Key for self.data dictionary and name-attribute for the saved CloudnetArray object.
            units: Units attribute for the CloudnetArray object.
            ignore_mask: If true, always writes an ordinary numpy array.

        Raises:
            RuntimeError: No variable found.

        """
        for name in possible_names:
            if name in self.dataset.variables:
                array = self.dataset.variables[name]
                if ignore_mask is True:
                    array = np.array(array)
                self.append_data(array, key, units=units)
                return
        raise RuntimeError("Missing variable in the input file.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
