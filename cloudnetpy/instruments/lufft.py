"""Module with a class for Lufft chm15k ceilometer."""
from typing import Union, List, Optional
import logging
import netCDF4
import numpy as np
from cloudnetpy.instruments.ceilometer import Ceilometer
from cloudnetpy import utils


class LufftCeilo(Ceilometer):
    """Class for Lufft chm15k ceilometer."""
    def __init__(self, file_name: str, date: Optional[str] = None):
        super().__init__(file_name)
        self._expected_date = date
        self.model = 'Lufft CHM15k'
        self.dataset = netCDF4.Dataset(self.file_name)
        self.variables = self.dataset.variables
        self.noise_params = (70, 2e-14, 0.3e-6, (1e-9, 4e-9))
        self.wavelength = 1064

    def read_ceilometer_file(self, calibration_factor: Optional[float] = None) -> None:
        """Reads data and metadata from Jenoptik netCDF file."""
        self.range = self._calc_range()
        self.backscatter = self._calibrate_backscatter(calibration_factor)
        self.time = self._fetch_time()
        self.date = self._read_date()
        self.metadata = self._read_metadata()

    def _calc_range(self) -> np.ndarray:
        """Assumes 'range' means the upper limit of range gate."""
        ceilo_range = self._getvar('range')
        return ceilo_range - utils.mdiff(ceilo_range)/2

    def _calibrate_backscatter(self, calibration_factor: Union[float, None]) -> np.ndarray:
        beta_raw = self._getvar('beta_raw')
        overlap_function = _get_overlap(self.range)
        beta_raw /= overlap_function
        if calibration_factor is None:
            logging.warning('Using default calibration factor for CHM15k')
            calibration_factor = 3e-12
        self.calibration_factor = calibration_factor
        beta_raw *= calibration_factor
        return beta_raw

    def _fetch_time(self) -> np.ndarray:
        time = self.variables['time'][:]
        ind = time.argsort()
        time = time[ind]
        self.backscatter = self.backscatter[ind, :]
        if self._expected_date is not None:
            epoch = utils.get_epoch(self.variables['time'].units)
            valid_ind = []
            for ind, timestamp in enumerate(time):
                date = '-'.join(utils.seconds2date(timestamp, epoch)[:3])
                if date == self._expected_date:
                    valid_ind.append(ind)
            if not valid_ind:
                raise ValueError('Error: CHM15k date differs from expected.')
            time = time[valid_ind]
            self.backscatter = self.backscatter[valid_ind, :]
        return utils.seconds2hours(time)

    def _read_date(self) -> List[str]:
        return [str(self.dataset.year),
                str(self.dataset.month).zfill(2),
                str(self.dataset.day).zfill(2)]

    def _getvar(self, *args) -> Union[np.ndarray, float, None]:
        for arg in args:
            if arg in self.variables:
                var = self.variables[arg]
                return var[0] if utils.isscalar(var) else var[:]
        return None

    def _read_metadata(self) -> dict:
        return {'tilt_angle': self._getvar('zenith')}


def _get_overlap(range_ceilo: np.ndarray,
                 params: Optional[tuple] = (0, 1)) -> np.ndarray:
    """Returns approximative overlap function."""
    return utils.array_to_probability(range_ceilo, *params)
