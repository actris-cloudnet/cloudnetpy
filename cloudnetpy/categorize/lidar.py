"""Lidar module, containing the :class:`Lidar` class."""
import logging
import numpy as np
import numpy.ma as ma
from cloudnetpy.categorize.datasource import DataSource


class Lidar(DataSource):
    """Lidar class, child of DataSource.

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

    def remove_low_level_outliers(self):
        n_cleaned_profiles = 0
        for ind, prof in enumerate(self.data['beta']):
            lower_part = prof[:15]
            values = lower_part[~lower_part.mask].data
            try:
                q1 = np.quantile(values, 0.25)
                q3 = np.quantile(values, 0.75)
            except IndexError:
                continue
            iqr = q3 - q1
            threshold = q3 + (1.5 * iqr)
            outliers = values > threshold
            if np.any(outliers):
                highest_outlier = max(np.where(outliers)[0]) + 3
                self.data['beta'][:][ind, :highest_outlier] = ma.masked
                n_cleaned_profiles += 1
        if n_cleaned_profiles > 0:
            logging.info(f'Cleaned {n_cleaned_profiles} profiles from low level lidar artifacts')

    def _add_meta(self) -> None:
        self.append_data(float(self.getvar('wavelength')), 'lidar_wavelength')
        self.append_data(0.5, 'beta_error')
        self.append_data(3, 'beta_bias')
