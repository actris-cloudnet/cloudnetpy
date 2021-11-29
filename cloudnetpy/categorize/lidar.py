"""Lidar module, containing the :class:`Lidar` class."""
import logging
import numpy as np
import numpy.ma as ma
from cloudnetpy.categorize.datasource import DataSource
from cloudnetpy.utils import interpolate_2d_nearest


class Lidar(DataSource):
    """Lidar class, child of DataSource.

    Args:
        full_path: Cloudnet Level 1 lidar netCDF file.

    """
    def __init__(self, full_path: str):
        super().__init__(full_path)
        self.append_data(self.getvar('beta'), 'beta')
        self._add_meta()

    def rebin_to_grid(self, time_new: np.ndarray, height_new: np.ndarray) -> None:
        """Rebins lidar data in time and height using mean.

        Args:
            time_new: 1-D target time array (fraction hour).
            height_new: 1-D target height array (m).

        """
        self.data['beta'].rebin_data(self.time, time_new, self.height, height_new)

    def interpolate_to_grid(self, time_new: np.ndarray, height_new: np.ndarray):
        """Interpolate beta using nearest neighbor."""
        max_height = 100  # m
        max_time = 1  # min
        max_time /= 60
        beta_interpolated = interpolate_2d_nearest(self.time, self.height, self.data['beta'][:],
                                                   time_new, height_new)
        bad_time_indices = _get_bad_indices(self.time, time_new, max_time)
        bad_height_indices = _get_bad_indices(self.height, height_new, max_height)
        if bad_time_indices:
            logging.warning(f'Unable to interpolate lidar for {len(bad_time_indices)} time steps')
        beta_interpolated[bad_time_indices, :] = ma.masked
        if bad_height_indices:
            logging.warning(f'Unable to interpolate lidar for {len(bad_height_indices)} altitudes')
        beta_interpolated[:, bad_height_indices] = ma.masked
        self.data['beta'].data = beta_interpolated

    def _add_meta(self) -> None:
        self.append_data(float(self.getvar('wavelength')), 'lidar_wavelength')
        self.append_data(0.5, 'beta_error')
        self.append_data(3.0, 'beta_bias')


def _get_bad_indices(original_grid: np.ndarray, new_grid: np.ndarray, threshold: float):
    indices = []
    min_original = min(original_grid)
    max_original = max(original_grid)
    for ind, value in enumerate(new_grid):
        if value < min_original or value > max_original:
            continue
        diffu = np.abs(original_grid - value)
        distance = diffu[diffu.argmin()]
        if distance > threshold:
            indices.append(ind)
    return indices
