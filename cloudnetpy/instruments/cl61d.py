"""Module with a class for Lufft chm15k ceilometer."""
from typing import Optional, Tuple
import logging
import netCDF4
import numpy as np
import scipy.ndimage
from cloudnetpy.instruments.ceilometer import NoiseParam, NoisyData, calc_sigma_units
from cloudnetpy.instruments.nc_lidar import NcLidar


class Cl61d(NcLidar):
    """Class for Vaisala CL61d ceilometer."""

    noise_param = NoiseParam(n_gates=100)

    def __init__(self, file_name: str, expected_date: Optional[str] = None):
        super().__init__(self.noise_param)
        self.file_name = file_name
        self.expected_date = expected_date
        self.model = 'Vaisala CL61d'
        self.wavelength = 910.55

    def read_ceilometer_file(self, calibration_factor: Optional[float] = None) -> None:
        """Reads data and metadata from concatenated Vaisala CL61d netCDF file."""
        self.dataset = netCDF4.Dataset(self.file_name)
        self._fetch_tilt_angle('zenith', default=3)
        self._fetch_range(reference='lower')
        self._fetch_lidar_variables(calibration_factor)
        self._fetch_time_and_date()
        self.dataset.close()

    def _fetch_lidar_variables(self, calibration_factor: Optional[float] = None) -> None:
        beta_raw = self.dataset.variables['beta_att'][:]
        if calibration_factor is None:
            logging.warning('Using default calibration factor')
            calibration_factor = 1
        beta_raw *= calibration_factor
        self.data['calibration_factor'] = calibration_factor
        self.data['beta_raw'] = beta_raw
        for key in ('p_pol', 'x_pol'):
            self.data[key] = self.dataset.variables[key][:]

    def calc_depol(self) -> Tuple[np.ndarray, np.ndarray]:
        """Converts raw depolarisation to noise-screened depolarisation."""
        snr_limit = 4
        noisy_data = NoisyData(self.data, self.noise_param)
        sigma = calc_sigma_units(self.data['time'], self.data['range'])
        x_pol = noisy_data.screen_data(self.data['x_pol'], keep_negative=True, snr_limit=snr_limit)
        depol = x_pol / self.data['p_pol']
        p_pol_smooth = scipy.ndimage.filters.gaussian_filter(self.data['p_pol'], sigma)
        x_pol_smooth = scipy.ndimage.filters.gaussian_filter(self.data['x_pol'], sigma)
        x_pol_smooth = noisy_data.screen_data(x_pol_smooth, is_smoothed=True, snr_limit=snr_limit)
        depol_smooth = x_pol_smooth / p_pol_smooth
        return depol, depol_smooth
