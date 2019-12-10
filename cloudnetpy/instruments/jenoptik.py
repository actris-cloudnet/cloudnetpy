"""Module with a class for Jenoptik ceilometer."""
from collections import namedtuple
import netCDF4
import numpy as np
import numpy.ma as ma
from cloudnetpy.instruments.ceilometer import Ceilometer
from cloudnetpy import utils

instrument_info = namedtuple('instrument_info',
                             ['calibration_factor',
                              'overlap_function_params',
                              'is_range_corrected'])

CEILOMETER_INFO = {
    'punta-arenas': instrument_info(
        calibration_factor=1e-12,
        overlap_function_params=(0, 1),
        is_range_corrected=True),
    'mace-head': instrument_info(
        calibration_factor=4.5e-11,
        overlap_function_params=(500, 200),
        is_range_corrected=False)
}


class JenoptikCeilo(Ceilometer):
    """Class for Jenoptik chm15k ceilometer."""
    def __init__(self, file_name, site_name):
        super().__init__(file_name)
        self.model = 'chm15k'
        self.dataset = netCDF4.Dataset(self.file_name)
        self.variables = self.dataset.variables
        self.noise_params = (70, 2e-14, 0.3e-6, (1e-9, 4e-9))
        self.calibration_info = _read_calibration_info(site_name)

    def read_ceilometer_file(self):
        """Reads data and metadata from Jenoptik netCDF file."""
        self.range = self._calc_range()
        self.time = self._convert_time()
        self.date = self._read_date()
        self.backscatter = self._convert_backscatter()
        self.metadata = self._read_metadata()

    def _calc_range(self):
        """Assumes 'range' means the upper limit of range gate."""
        ceilo_range = self._getvar('range')
        return ceilo_range - utils.mdiff(ceilo_range)/2

    def _convert_time(self):
        time = self.variables['time']
        try:
            assert all(np.diff(time) > 0)
        except AssertionError:
            raise RuntimeError('Inconsistent ceilometer time stamps.')
        if max(time) > 24:
            time = utils.seconds2hours(time)
        return time

    def _read_date(self):
        return [self.dataset.year, self.dataset.month, self.dataset.day]

    def _convert_backscatter(self):
        """Steps to convert Jenoptik SNR to raw beta."""
        beta_raw = self._getvar('beta_raw')
        data_std = self._getvar('stddev')
        normalised_apd = self._get_nn()
        beta_raw *= utils.transpose(data_std / normalised_apd)
        if not self.calibration_info.is_range_corrected:
            beta_raw *= self.range ** 2
        overlap_function = _get_overlap(self.range, self.calibration_info)
        beta_raw /= overlap_function
        beta_raw *= self.calibration_info.calibration_factor
        return beta_raw

    def _get_nn(self):
        """TODO: Should be checked."""
        nn1 = self._getvar('nn1', 'NN1')
        median_nn1 = ma.median(nn1)
        if 120 < median_nn1 < 160:
            step_factor, reference, scale = 1.24, 140, 5
        elif 3200 < median_nn1 < 4000:
            step_factor, reference, scale = 1.035, 3685, 1
        else:
            return 1
        return step_factor ** (-(nn1-reference) / scale)

    def _getvar(self, *args):
        """Reads data of variable (array or scalar) from netcdf-file."""
        for arg in args:
            if arg in self.variables:
                var = self.variables[arg]
                return var[0] if utils.isscalar(var) else var[:]

    def _read_metadata(self):
        meta = {'tilt_angle': self._getvar('zenith')}
        return meta


def _get_overlap(range_ceilo, calibration_info):
    """Approximative overlap function."""
    params = calibration_info.overlap_function_params
    return utils.array_to_probability(range_ceilo, *params)


def _read_calibration_info(site_name):
    if 'punta' in site_name.lower():
        return CEILOMETER_INFO['punta-arenas']
    elif 'mace' in site_name.lower():
        return CEILOMETER_INFO['mace-head']
