import logging
from typing import Optional, Union

import netCDF4
import numpy as np
from numpy import ma

from cloudnetpy import CloudnetArray, utils


class CloudnetInstrument:
    def __init__(self):
        self.dataset: Optional[netCDF4.Dataset] = None
        self.time: np.ndarray = np.array([])
        self.site_meta: dict = {}
        self.data: dict = {}

    def add_site_geolocation(self) -> None:
        for key in ("latitude", "longitude", "altitude"):
            value = None
            # User-supplied:
            if key in self.site_meta:
                value = self.site_meta[key]
            # From source global attributes (MIRA):
            elif isinstance(self.dataset, netCDF4.Dataset) and hasattr(
                self.dataset, key.capitalize()
            ):
                value = float(getattr(self.dataset, key.capitalize()).split()[0])
            # From source data (BASTA / RPG):
            elif isinstance(self.dataset, netCDF4.Dataset) and key in self.dataset.variables:
                value = self.dataset.variables[key][:]
            if value is not None:
                value = float(ma.mean(value))
                self.data[key] = CloudnetArray(value, key)

    def add_height(self) -> None:
        try:
            zenith_angle = ma.median(self.data["zenith_angle"].data)
        except RuntimeError:
            logging.warning("Assuming 0 deg zenith_angle")
            zenith_angle = 0
        height = utils.range_to_height(self.data["range"].data, zenith_angle)
        height += self.data["altitude"].data
        height = np.array(height)
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

    def screen_time_indices(self, valid_indices: Union[list, np.ndarray]) -> None:
        time = self._get_time()
        n_time = len(time)
        for cloudnet_array in self.data.values():
            array = cloudnet_array.data
            if not utils.isscalar(array) and array.shape[0] == n_time:
                if array.ndim == 1:
                    cloudnet_array.data = array[valid_indices]
                elif array.ndim == 2:
                    cloudnet_array.data = array[valid_indices, :]
                elif array.ndim == 3:
                    cloudnet_array.data = array[valid_indices, :]
        if self.time.size > 0:
            self.time = self.time[valid_indices]

    def _get_time(self) -> np.ndarray:
        try:
            return self.data["time"].data[:]
        except KeyError:
            return self.time
