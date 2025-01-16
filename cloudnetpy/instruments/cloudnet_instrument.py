import logging

import netCDF4
import numpy as np
from numpy import ma

from cloudnetpy import utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.exceptions import ValidTimeStampError


class CloudnetInstrument:
    def __init__(self):
        self.dataset: netCDF4.Dataset
        self.time: np.ndarray = np.array([])
        self.site_meta: dict = {}
        self.data: dict = {}
        self.serial_number: str | None = None

    def add_site_geolocation(self) -> None:
        for key in ("latitude", "longitude", "altitude"):
            value = None
            # User-supplied:
            if key in self.site_meta:
                value = self.site_meta[key]
            # From source global attributes (MIRA):
            elif (
                hasattr(self, "dataset")
                and isinstance(self.dataset, netCDF4.Dataset)
                and hasattr(
                    self.dataset,
                    key.capitalize(),
                )
            ):
                value = self.parse_global_attribute_numeral(key.capitalize())
            # From source data (BASTA / RPG):
            elif (
                hasattr(self, "dataset")
                and isinstance(self.dataset, netCDF4.Dataset)
                and key in self.dataset.variables
            ):
                value = self.dataset.variables[key][:]
            if value is not None:
                value = float(ma.mean(value))
                self.data[key] = CloudnetArray(value, key)

    def parse_global_attribute_numeral(self, key: str) -> float:
        new_str = ""
        for char in getattr(self.dataset, key):
            if char.isdigit() or char == ".":
                new_str += char
        return float(new_str)

    def add_height(self) -> None:
        zenith_angle = self._get_zenith_angle()
        if zenith_angle is None:
            logging.warning("Assuming 0 deg zenith_angle")
            zenith_angle = 0
        height = utils.range_to_height(self.data["range"].data, zenith_angle)
        height += self.data["altitude"].data
        self.data["height"] = CloudnetArray(height, "height")

    def linear_to_db(self, variables_to_log: tuple) -> None:
        """Changes linear units to logarithmic."""
        for name in variables_to_log:
            self.data[name].lin2db()

    def remove_duplicate_timestamps(self) -> None:
        time = self._get_time()
        _, ind = np.unique(time, return_index=True)
        self.screen_time_indices(ind)

    def sort_timestamps(self) -> None:
        time = self._get_time()
        ind = time.argsort()
        self.screen_time_indices(ind)

    def screen_time_indices(self, valid_indices: list | np.ndarray) -> None:
        time = self._get_time()
        n_time = len(time)
        if len(valid_indices) == 0 or (
            isinstance(valid_indices, np.ndarray)
            and valid_indices.dtype == np.bool_
            and valid_indices.shape == time.shape
            and not np.any(valid_indices)
        ):
            msg = "All timestamps screened"
            raise ValidTimeStampError(msg)
        for cloudnet_array in self.data.values():
            array = cloudnet_array.data
            if not utils.isscalar(array) and array.shape[0] == n_time:
                match array.ndim:
                    case 1:
                        cloudnet_array.data = array[valid_indices]
                    case 2:
                        cloudnet_array.data = array[valid_indices, :]
                    case 3:
                        cloudnet_array.data = array[valid_indices, :, :]
        if self.time.size > 0:
            self.time = self.time[valid_indices]

    def _get_time(self) -> np.ndarray:
        try:
            return self.data["time"].data[:]
        except KeyError:
            return self.time

    def _get_zenith_angle(self) -> float | None:
        if "zenith_angle" not in self.data or self.data["zenith_angle"].data.size == 0:
            return None
        zenith_angle = ma.median(self.data["zenith_angle"].data)
        if np.isnan(zenith_angle) or zenith_angle is ma.masked:
            return None
        return zenith_angle


class CSVFile(CloudnetInstrument):
    def __init__(self, site_meta: dict):
        super().__init__()
        self.site_meta = site_meta
        self._data: dict = {}

    def add_date(self) -> None:
        dt = self._data["time"][0]
        self.date = dt.strftime("%Y %m %d").split()

    def add_data(self) -> None:
        for key, value in self._data.items():
            parsed = (
                utils.datetime2decimal_hours(value)
                if key == "time"
                else ma.array(value)
            )
            self.data[key] = CloudnetArray(parsed, key)

    def normalize_rainfall_amount(self) -> None:
        if "rainfall_amount" in self.data:
            amount = self.data["rainfall_amount"][:]
            offset = 0
            for i in range(1, len(amount)):
                if amount[i] + offset < amount[i - 1]:
                    offset += amount[i - 1]
                amount[i] += offset
            amount -= amount[0]
            self.data["rainfall_amount"].data = amount
