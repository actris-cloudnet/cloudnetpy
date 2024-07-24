"""Lidar module, containing the :class:`Lidar` class."""

import logging
from typing import Literal

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

    def interpolate_to_grid(
        self, time_new: np.ndarray, height_new: np.ndarray
    ) -> list[int]:
        """Interpolate beta using nearest neighbor."""
        max_height = 100  # m
        max_time = 1 / 60  # min -> fraction hour

        if self.height is None:
            msg = "Unable to interpolate lidar: no height information"
            raise RuntimeError(msg)

        # Interpolate beta to new grid but ignore profiles that are completely masked
        beta = self.data["beta"][:]
        indices = [ind for ind, b in enumerate(beta) if ma.all(b) is not ma.masked]
        beta_interp = interpolate_2d_nearest(
            self.time[indices],
            self.height,
            beta[indices, :],
            time_new,
            height_new,
        )
        # Mask data points that are too far from the original grid
        time_gap_ind = get_gap_ind(self.time[indices], time_new, max_time)
        height_gap_ind = get_gap_ind(self.height, height_new, max_height)
        self._mask_profiles(beta_interp, time_gap_ind, "time")
        self._mask_profiles(beta_interp, height_gap_ind, "height")
        self.data["beta"].data = beta_interp
        return time_gap_ind

    @staticmethod
    def _mask_profiles(
        data: ma.MaskedArray, ind: list[int], dim: Literal["time", "height"]
    ) -> None:
        prefix = f"Unable to interpolate lidar for {len(ind)}"
        if dim == "time" and ind:
            logging.warning("%s time steps", prefix)
            data[ind, :] = ma.masked
        elif dim == "height" and ind:
            logging.warning("%s altitudes", prefix)
            data[:, ind] = ma.masked

    def _add_meta(self) -> None:
        self.append_data(float(self.getvar("wavelength")), "lidar_wavelength")
        self.append_data(0.5, "beta_error")
        self.append_data(3.0, "beta_bias")


def get_gap_ind(grid: np.ndarray, new_grid: np.ndarray, threshold: float) -> list[int]:
    return [
        ind
        for ind, value in enumerate(new_grid)
        if np.min(np.abs(grid - value)) > threshold
    ]
