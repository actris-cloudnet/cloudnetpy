"""Module with a class for Jenoptik ceilometer."""
from typing import Union, List
from collections import namedtuple
import netCDF4
import numpy as np
from cloudnetpy.instruments.ceilometer import Ceilometer
from cloudnetpy import utils

instrument_info = namedtuple('instrument_info',
                             ['calibration_factor',
                              'overlap_function_params',
                              'is_range_corrected'])

# TODO: should be a separate config file or accessible over http api
CEILOMETER_INFO = {
    'punta-arenas': instrument_info(
        calibration_factor=1e-12,
        overlap_function_params=None,
        is_range_corrected=True),
    'mace-head': instrument_info(
        calibration_factor=5.2e-15,
        overlap_function_params=(500, 200),
        is_range_corrected=False),
    'bucharest': instrument_info(
        calibration_factor=5e-12,
        overlap_function_params=None,
        is_range_corrected=True),
    'granada': instrument_info(
        calibration_factor=5.2e-12,
        overlap_function_params=None,
        is_range_corrected=True),
    'lindenberg': instrument_info(
        calibration_factor=2.5e-11,
        overlap_function_params=None,
        is_range_corrected=True),
    'palaiseau': instrument_info(
        calibration_factor=3.6e-11,
        overlap_function_params=None,
        is_range_corrected=True),
}


class JenoptikCeilo(Ceilometer):
    """Class for Jenoptik chm15k ceilometer."""
    def __init__(self, file_name, site_name):
        super().__init__(file_name)
        self.model = 'Lufft CHM15k'
        self.dataset = netCDF4.Dataset(self.file_name)
        self.variables = self.dataset.variables
        self.noise_params = (70, 2e-14, 0.3e-6, (1e-9, 4e-9))
        self.calibration_info = _read_calibration_info(site_name)

    def read_ceilometer_file(self) -> None:
        """Reads data and metadata from Jenoptik netCDF file."""
        self.range = self._calc_range()
        self.time = self._convert_time()
        self.date = self._read_date()
        self.backscatter = self._convert_backscatter()
        self.metadata = self._read_metadata()

    def _calc_range(self) -> np.ndarray:
        """Assumes 'range' means the upper limit of range gate."""
        ceilo_range = self._getvar('range')
        return ceilo_range - utils.mdiff(ceilo_range)/2

    def _convert_time(self) -> np.ndarray:
        time = self.variables['time']
        try:
            assert all(np.diff(time) > 0)
        except AssertionError:
            raise RuntimeError('Inconsistent ceilometer time stamps.')
        if max(time) > 24:
            time = utils.seconds2hours(time)
        return time

    def _read_date(self) -> List[str]:
        return [str(self.dataset.year),
                str(self.dataset.month).zfill(2),
                str(self.dataset.day).zfill(2)]

    def _convert_backscatter(self) -> np.ndarray:
        """Steps to convert Jenoptik SNR to raw beta."""
        beta_raw = self._getvar('beta_raw')
        if not self.calibration_info.is_range_corrected:
            beta_raw *= self.range ** 2
        overlap_function = _get_overlap(self.range, self.calibration_info)
        beta_raw /= overlap_function
        beta_raw *= self.calibration_info.calibration_factor
        return beta_raw

    def _getvar(self, *args) -> Union[np.ndarray, float, None]:
        """Reads data of variable (array or scalar) from netcdf-file."""
        for arg in args:
            if arg in self.variables:
                var = self.variables[arg]
                return var[0] if utils.isscalar(var) else var[:]
        return None

    def _read_metadata(self) -> dict:
        return {'tilt_angle': self._getvar('zenith')}


def _get_overlap(range_ceilo: np.ndarray, calibration_info: instrument_info) -> np.ndarray:
    """Approximative overlap function."""
    params = calibration_info.overlap_function_params or (0, 1)
    return utils.array_to_probability(range_ceilo, *params)


def _read_calibration_info(site_name: str) -> instrument_info:
    if 'punta' in site_name.lower():
        return CEILOMETER_INFO['punta-arenas']
    if 'mace' in site_name.lower():
        return CEILOMETER_INFO['mace-head']
    return CEILOMETER_INFO[site_name.lower()]
