"""Datasource module, containing the :class:`DataSource` class."""

import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone

import netCDF4
import numpy as np

from cloudnetpy import utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.exceptions import ValidTimeStampError


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
    source_type: str

    def __init__(self, full_path: os.PathLike | str, *, radar: bool = False):
        self.filename = os.path.basename(full_path)
        self.dataset = netCDF4.Dataset(full_path)
        self.source = getattr(self.dataset, "source", "")
        self.time: np.ndarray = self._init_time()
        self.altitude = self._init_altitude()
        self.height = self._init_height()
        self.data: dict = {}
        self._is_radar = radar

    def getvar(self, *args) -> np.ndarray:
        """Returns data array from the source file variables.

        Returns just the data (and no attributes) from the original
            variables dictionary, fetched from the input netCDF file.

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
        msg = f"Missing variable {args[0]} in the input file."
        raise RuntimeError(msg)

    def append_data(
        self,
        variable: netCDF4.Variable | np.ndarray | float,
        key: str,
        name: str | None = None,
        units: str | None = None,
        dtype: str | None = None,
    ) -> None:
        """Adds new CloudnetVariable or RadarVariable into `data` attribute.

        Args:
            variable: netCDF variable or data array to be added.
            key: Key used with *variable* when added to `data`
                attribute (dictionary).
            name: CloudnetArray.name attribute. Default value is *key*.
            units: CloudnetArray.units attribute.
            dtype: CloudnetArray.data_type attribute.

        """
        self.data[key] = CloudnetArray(variable, name or key, units, data_type=dtype)

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
            datetime.strptime(f"{year}{month}{day}", "%Y%m%d").replace(
                tzinfo=timezone.utc,
            )

        except (AttributeError, ValueError) as read_error:
            msg = "Missing or invalid date in global attributes."
            raise RuntimeError(msg) from read_error
        return [year, month, day]

    def close(self) -> None:
        """Closes the open file."""
        self.dataset.close()

    @staticmethod
    def to_m(var: netCDF4.Variable) -> np.ndarray:
        """Converts km to m."""
        alt = var[:]
        if var.units == "km":
            alt *= 1000
        elif var.units not in ("m", "meters"):
            msg = f"Unexpected unit: {var.units}"
            raise ValueError(msg)
        return alt

    @staticmethod
    def to_km(var: netCDF4.Variable) -> np.ndarray:
        """Converts m to km."""
        alt = var[:]
        if var.units == "m":
            alt /= 1000
        elif var.units != "km":
            msg = f"Unexpected unit: {var.units}"
            raise ValueError(msg)
        return alt

    def _init_time(self) -> np.ndarray:
        time = self.getvar("time")
        if len(time) == 0:
            msg = "Empty time vector"
            raise ValidTimeStampError(msg)
        if max(time) > 25:
            logging.debug("Assuming time as seconds, converting to fraction hour")
            time = utils.seconds2hours(time)
        return time

    def _init_altitude(self) -> float | None:
        """Returns altitude of the instrument (m)."""
        if "altitude" in self.dataset.variables:
            var = self.dataset.variables["altitude"]
            if utils.is_all_masked(var[:]):
                return None
            altitude_above_sea = self.to_m(var)
            return float(
                altitude_above_sea
                if utils.isscalar(altitude_above_sea)
                else np.mean(altitude_above_sea),
            )
        return None

    def _init_height(self) -> np.ndarray | None:
        """Returns height array above mean sea level (m)."""
        if "height" in self.dataset.variables:
            return self.to_m(self.dataset.variables["height"])
        if "range" in self.dataset.variables and self.altitude is not None:
            range_instrument = self.to_m(self.dataset.variables["range"])
            return np.array(range_instrument + self.altitude)
        return None

    def _variables_to_cloudnet_arrays(self, keys: tuple) -> None:
        """Transforms netCDF4-variables into CloudnetArrays.

        Args:
            keys: netCDF4-variables to be converted. The results
                are saved in *self.data* dictionary with *fields*
                strings as keys.

        Notes:
            The attributes of the variables are not copied. Just the data.

        """
        for key in keys:
            self.append_data(self.dataset.variables[key], key)

    def _unknown_variable_to_cloudnet_array(
        self,
        possible_names: tuple,
        key: str,
        units: str | None = None,
        *,
        ignore_mask: bool = False,
    ) -> None:
        """Transforms single netCDF4 variable into CloudnetArray.

        Args:
            possible_names: Tuple of strings containing the possible
                names of the variable in the input NetCDF file.
            key: Key for self.data dictionary and name-attribute
                for the saved CloudnetArray object.
            units: Units attribute for the CloudnetArray object.
            ignore_mask: If true, always writes an ordinary numpy array.

        Raises:
            RuntimeError: No variable found.

        """
        for name in possible_names:
            if name in self.dataset.variables:
                array: netCDF4.Variable | np.ndarray = self.dataset.variables[name]
                if ignore_mask is True:
                    array = np.array(array)
                self.append_data(array, key, units=units)
                return
        msg = f"Missing variable {possible_names[0]} in the input file."
        raise RuntimeError(msg)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
