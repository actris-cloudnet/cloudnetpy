import logging
from typing import Union

import numpy as np
from numpy import ma

from cloudnetpy import CloudnetArray, utils


class CloudnetInstrument:

    data: dict
    time: np.ndarray

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
        if hasattr(self, "time"):
            self.time = self.time[valid_indices]

    def _get_time(self) -> np.ndarray:
        try:
            return self.data["time"].data[:]
        except KeyError:
            return self.time
