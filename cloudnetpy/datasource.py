"""Datasource module, containing the :class:`DataSource` class."""

import datetime
import logging
import os
from collections.abc import Callable
from os import PathLike
from types import TracebackType

import netCDF4
import numpy as np
import numpy.typing as npt
from typing_extensions import Self

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

    def __init__(self, full_path: PathLike | str, *, radar: bool = False) -> None:
        self.filename = os.path.basename(full_path)
        self.dataset = netCDF4.Dataset(full_path)
        self.source = getattr(self.dataset, "source", "")
        self.instrument_pid = getattr(self.dataset, "instrument_pid", "")
        self.time: npt.NDArray = self._init_time()
        self.altitude = self._init_altitude()
        self.height = self._init_height()
        self.height_agl = (
            self.height - self.altitude
            if self.height is not None and self.altitude is not None
            else None
        )
        self.data: dict = {}
        self._is_radar = radar

    def getvar(self, *args: str) -> npt.NDArray:
        """Returns data array from the source file variables.

        Returns just the data (and no attributes) from the original
            variables dictionary, fetched from the input netCDF file.

        Args:
            *args: possible names of the variable. The first match is returned.

        Returns:
            ndarray: The actual data.

        Raises:
             KeyError: The variable is not found.

        """
        for arg in args:
            if arg in self.dataset.variables:
                return self.dataset.variables[arg][:]
        msg = f"Missing variable {args[0]} in the input file."
        raise KeyError(msg)

    def append_data(
        self,
        variable: netCDF4.Variable | npt.NDArray | float,
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

    def get_date(self) -> datetime.date:
        """Returns date components.

        Returns:
            date object

        Raises:
             RuntimeError: Not found or invalid date.

        """
        try:
            year = int(self.dataset.year)
            month = int(self.dataset.month)
            day = int(self.dataset.day)
            return datetime.date(year, month, day)
        except (AttributeError, ValueError) as read_error:
            msg = "Missing or invalid date in global attributes."
            raise RuntimeError(msg) from read_error

    def close(self) -> None:
        """Closes the open file."""
        self.dataset.close()

    @staticmethod
    def to_m(var: netCDF4.Variable) -> npt.NDArray:
        """Converts km to m."""
        alt = var[:]
        if var.units == "km":
            alt *= 1000
        elif var.units not in ("m", "meters"):
            msg = f"Unexpected unit: {var.units}"
            raise ValueError(msg)
        return alt

    def _init_time(self) -> npt.NDArray:
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

    def _init_height(self) -> npt.NDArray | None:
        """Returns height array above mean sea level (m)."""
        if "height" in self.dataset.variables:
            return self.to_m(self.dataset.variables["height"])
        if "range" in self.dataset.variables and self.altitude is not None:
            range_instrument = self.to_m(self.dataset.variables["range"])
            return np.array(range_instrument + self.altitude)
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()
