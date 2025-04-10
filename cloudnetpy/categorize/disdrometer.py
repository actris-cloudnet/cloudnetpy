"""Mwr module, containing the :class:`Mwr` class."""

import logging

import numpy as np
from numpy import ma
from scipy.interpolate import interp1d

from cloudnetpy.categorize.lidar import get_gap_ind
from cloudnetpy.datasource import DataSource
from cloudnetpy.exceptions import DisdrometerDataError


class Disdrometer(DataSource):
    """Disdrometer class, child of DataSource.

    Args:
    ----
         full_path: Cloudnet Level 1b disdrometer file.

    """

    def __init__(self, full_path: str):
        super().__init__(full_path)
        self._init_rainfall_rate()

    def interpolate_to_grid(self, time_grid: np.ndarray) -> None:
        for key, array in self.data.items():
            self.data[key].data = self._interpolate(array.data, time_grid)

    def _init_rainfall_rate(self) -> None:
        keys = ("rainfall_rate", "n_particles")
        for key in keys:
            if key not in self.dataset.variables:
                msg = f"variable {key} is missing"
                raise DisdrometerDataError(msg)
            self.append_data(self.dataset.variables[key][:], key)

    def _interpolate(self, y: ma.MaskedArray, x_new: np.ndarray) -> np.ndarray:
        time = self.time
        mask = ma.getmask(y)
        if mask is not ma.nomask:
            if np.all(mask):
                return ma.masked_all(x_new.shape)
            not_masked = ~mask
            y = y[not_masked]
            time = time[not_masked]
        fun = interp1d(time, y, fill_value="extrapolate")
        interpolated_array = ma.array(fun(x_new))
        max_time = 1 / 60  # min -> fraction hour
        mask_ind = get_gap_ind(time, x_new, max_time)

        if len(mask_ind) > 0:
            msg = f"Unable to interpolate disdrometer for {len(mask_ind)} time steps"
            logging.warning(msg)
            interpolated_array[mask_ind] = ma.masked

        return interpolated_array
