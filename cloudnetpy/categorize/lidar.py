"""Lidar module, containing the :class:`Lidar` class."""
from cloudnetpy.categorize import ProfileDataSource


class Lidar(ProfileDataSource):
    """Lidar class, child of ProfileDataSource.

    Args:
        lidar_file (str): File name of the calibrated lidar file.

    """
    def __init__(self, lidar_file):
        super().__init__(lidar_file)
        self._unknown_to_cloudnet(('beta', 'beta_smooth'), 'beta')
        self._add_meta()

    def rebin_to_grid(self, time_new, height_new):
        """Rebins lidar data in time and height using mean.

        Args:
            time_new (ndarray): 1-D target time array (fraction hour).
            height_new (ndarray): 1-D target height array (m).

        """
        self.data['beta'].rebin_data(self.time, time_new, self.height,
                                     height_new)

    def _add_meta(self):
        self.append_data(float(self.getvar('wavelength')), 'lidar_wavelength')
        self.append_data(0.5, 'beta_error')
        self.append_data(3, 'beta_bias')
