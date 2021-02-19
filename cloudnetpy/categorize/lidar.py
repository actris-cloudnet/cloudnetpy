"""Lidar module, containing the :class:`Lidar` class."""
import numpy as np
from cloudnetpy.categorize.datasource import ProfileDataSource


class Lidar(ProfileDataSource):
    """Lidar class, child of ProfileDataSource.

    Args:
        full_path: Cloudnet Level 1 lidar netCDF file.

    """
    def __init__(self, full_path: str):
        super().__init__(full_path)
        self._unknown_variable_to_cloudnet_array(('beta', 'beta_smooth'), 'beta')
        self._add_meta()

    def rebin_to_grid(self, time_new: np.ndarray, height_new: np.ndarray) -> None:
        """Rebins lidar data in time and height using mean.

        Args:
            time_new: 1-D target time array (fraction hour).
            height_new: 1-D target height array (m).

        """
        self.data['beta'].rebin_data(self.time, time_new, self.height, height_new)

    def _add_meta(self) -> None:
        self.append_data(float(self.getvar('wavelength')), 'lidar_wavelength')
        self.append_data(0.5, 'beta_error')
        self.append_data(3, 'beta_bias')
