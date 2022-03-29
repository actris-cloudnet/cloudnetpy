"""Lidar module, containing the :class:`Lidar` class."""
import logging

import numpy as np
from numpy import ma

from cloudnetpy.datasource import DataSource
from cloudnetpy.utils import interpolate_2d_nearest


class Lidar(DataSource):
    """Lidar class, child of DataSource.

    Args:
        full_path: Cloudnet Level 1 lidar netCDF file.

    """

    def __init__(self, full_path: str):
        super().__init__(full_path)
        self.append_data(self.getvar("beta"), "beta")
        self._add_meta()

    def interpolate_to_grid(self, time_new: np.ndarray, height_new: np.ndarray) -> list:
        """Interpolate beta using nearest neighbor."""
        max_height = 100.0  # m
        max_time = 1.0  # min

        # Remove completely masked profiles from the interpolation
        beta = self.data["beta"][:]
        indices = []
        for ind, b in enumerate(beta):
            if not ma.all(b) is ma.masked:
                indices.append(ind)
        assert self.height is not None
        beta_interpolated = interpolate_2d_nearest(
            self.time[indices], self.height, beta[indices, :], time_new, height_new
        )

        # Filter profiles and range gates having data gap
        max_time /= 60  # to fraction hour
        bad_time_indices = _get_bad_indices(self.time[indices], time_new, max_time)
        bad_height_indices = _get_bad_indices(self.height, height_new, max_height)
        if bad_time_indices:
            logging.warning(f"Unable to interpolate lidar for {len(bad_time_indices)} time steps")
        beta_interpolated[bad_time_indices, :] = ma.masked
        if bad_height_indices:
            logging.warning(f"Unable to interpolate lidar for {len(bad_height_indices)} altitudes")
        beta_interpolated[:, bad_height_indices] = ma.masked
        self.data["beta"].data = beta_interpolated
        return bad_time_indices

    def _add_meta(self) -> None:
        self.append_data(float(self.getvar("wavelength")), "lidar_wavelength")
        self.append_data(0.5, "beta_error")
        self.append_data(3.0, "beta_bias")


def _get_bad_indices(original_grid: np.ndarray, new_grid: np.ndarray, threshold: float) -> list:
    indices = []
    for ind, value in enumerate(new_grid):
        diffu = np.abs(original_grid - value)
        distance = diffu[diffu.argmin()]
        if distance > threshold:
            indices.append(ind)
    return indices
