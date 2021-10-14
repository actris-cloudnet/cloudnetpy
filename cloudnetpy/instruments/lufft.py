"""Module with a class for Lufft chm15k ceilometer."""
from typing import Optional
import logging
import netCDF4
import numpy as np
from cloudnetpy.instruments.ceilometer import NoiseParam
from cloudnetpy.instruments.nc_lidar import NcLidar
from cloudnetpy import utils


class LufftCeilo(NcLidar):
    """Class for Lufft chm15k ceilometer."""

    noise_param = NoiseParam(n_gates=70)

    def __init__(self, file_name: str, expected_date: Optional[str] = None):
        super().__init__(self.noise_param)
        self.file_name = file_name
        self.expected_date = expected_date
        self.model = 'Lufft CHM15k'
        self.wavelength = 1064

    def read_ceilometer_file(self, calibration_factor: Optional[float] = None) -> None:
        """Reads data and metadata from Jenoptik netCDF file."""
        self.dataset = netCDF4.Dataset(self.file_name)
        self._fetch_range(reference='upper')
        self._fetch_beta_raw(calibration_factor)
        self._fetch_time_and_date()
        self._fetch_tilt_angle('zenith')
        self.dataset.close()

    def _fetch_beta_raw(self, calibration_factor: Optional[float] = None) -> None:
        beta_raw = self.dataset.variables['beta_raw'][:]
        overlap_function = _get_overlap(self.data['range'])
        beta_raw /= overlap_function
        if calibration_factor is None:
            logging.warning('Using default calibration factor')
            calibration_factor = 3e-12
        beta_raw *= calibration_factor
        self.data['calibration_factor'] = calibration_factor
        self.data['beta_raw'] = beta_raw


def _get_overlap(range_ceilo: np.ndarray,
                 params: Optional[tuple] = (0, 1)) -> np.ndarray:
    """Returns approximative overlap function."""
    return utils.array_to_probability(range_ceilo, *params)
