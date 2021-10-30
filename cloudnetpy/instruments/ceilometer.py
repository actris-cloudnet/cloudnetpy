from typing import Tuple, Optional
import numpy as np
import numpy.ma as ma
import scipy.ndimage
from cloudnetpy import utils
from cloudnetpy.cloudnetarray import CloudnetArray


class NoiseParam:
    """Noise parameters. Values are weakly instrument-dependent."""
    def __init__(self,
                 variance: Optional[float] = 2e-14,
                 saturation: Optional[float] = 0.3e-6,
                 min_noise: Optional[float] = 1e-9,
                 min_noise_smooth: Optional[float] = 4e-9):
        self.variance = variance
        self.saturation = saturation
        self.min_noise = min_noise
        self.min_noise_smooth = min_noise_smooth


class Ceilometer:
    """Base class for all types of ceilometers and pollyxt."""

    def __init__(self, noise_param: Optional[NoiseParam] = NoiseParam()):
        self.noise_param = noise_param
        self.data = {}          # Need to contain 'beta_raw', 'range' and 'time'
        self.metadata = {}      # Need to contain 'date' as ('yyyy', 'mm', 'dd')
        self.model = ''
        self.wavelength = None
        self.expected_date = None

    def calc_screened_product(self,
                              array: np.ndarray,
                              snr_limit: Optional[int] = 5) -> np.ndarray:
        """Screens noise from lidar variable."""
        noisy_data = NoisyData(self.data, self.noise_param)
        array_screened = noisy_data.screen_data(array, snr_limit=snr_limit)
        return array_screened

    def calc_beta_smooth(self, beta: np.ndarray, snr_limit: Optional[int] = 5) -> np.ndarray:
        noisy_data = NoisyData(self.data, self.noise_param)
        beta_raw = ma.copy(self.data['beta_raw'])
        cloud_ind, cloud_values, cloud_limit = _estimate_clouds_from_beta(beta)
        beta_raw[cloud_ind] = cloud_limit
        sigma = calc_sigma_units(self.data['time'], self.data['range'])
        beta_raw_smooth = scipy.ndimage.filters.gaussian_filter(beta_raw, sigma)
        beta_raw_smooth[cloud_ind] = cloud_values
        beta_smooth = noisy_data.screen_data(beta_raw_smooth, is_smoothed=True, snr_limit=snr_limit)
        return beta_smooth

    def prepare_data(self, site_meta: dict):
        """Add common additional data / metadata and convert into CloudnetArrays."""
        zenith_angle = self.data['zenith_angle']
        self.data['height'] = np.array(site_meta['altitude']
                                       + utils.range_to_height(self.data['range'], zenith_angle))
        for key in ('time', 'range'):
            self.data[key] = np.array(self.data[key])
        assert self.wavelength is not None
        self.data['wavelength'] = float(self.wavelength)
        for key in ('latitude', 'longitude', 'altitude'):
            if key in site_meta:
                self.data[key] = float(site_meta[key])

    def prepare_metadata(self):
        """Add common global attributes."""
        assert self.model is not None
        self.metadata['source'] = self.model

    def get_date_and_time(self, epoch: tuple) -> None:
        if self.expected_date is not None:
            self.data = utils.screen_by_time(self.data, epoch, self.expected_date)
        self.metadata['date'] = utils.seconds2date(self.data['time'][0], epoch=epoch)[:3]
        self.data['time'] = utils.seconds2hours(self.data['time'])

    def remove_raw_data(self):
        keys = [key for key in self.data.keys() if 'raw' in key]
        for key in keys:
            del self.data[key]
        self.data.pop('x_pol', None)
        self.data.pop('p_pol', None)

    def data_to_cloudnet_arrays(self):
        for key, array in self.data.items():
            self.data[key] = CloudnetArray(array, key)

    def screen_depol(self):
        key = 'depolarisation'
        if key in self.data:
            self.data[key][self.data[key] <= 0] = ma.masked
            self.data[key][self.data[key] > 1] = ma.masked

    def add_snr_info(self, key: str, snr_limit: float):
        if key in self.data:
            self.data[key].comment += f' SNR threshold applied: {snr_limit}.'


class NoisyData:
    def __init__(self, data: dict, noise_param: NoiseParam):
        self.data = data
        self.noise_param = noise_param

    def screen_data(self,
                    data: np.array,
                    snr_limit: Optional[float] = 5,
                    is_smoothed: Optional[bool] = False,
                    keep_negative: Optional[bool] = False) -> np.ndarray:
        array = self._calc_range_uncorrected(data)
        array = self._screen_by_snr(array, is_smoothed, keep_negative, snr_limit)
        array = self._calc_range_corrected(array)
        return array

    def _screen_by_snr(self,
                       array: np.ndarray,
                       is_smoothed: bool,
                       keep_negative: bool,
                       snr_limit: float) -> np.ndarray:
        """Screens noise from range-uncorrected lidar variable."""
        if is_smoothed is True:
            noise_min = self.noise_param.min_noise_smooth
        else:
            noise_min = self.noise_param.min_noise
        noise = _estimate_background_noise(array, noise_min)
        array = self._reset_low_values_above_saturation(array)
        array = self._remove_noise(array, noise, keep_negative, snr_limit)
        return array

    def _reset_low_values_above_saturation(self, array: np.ndarray) -> np.ndarray:
        """Removes low values in saturated profiles above peak."""
        is_saturation = self._find_saturated_profiles()
        for saturated_profile in np.where(is_saturation)[0]:
            profile = array[saturated_profile, :]
            peak_ind = np.argmax(profile)
            alt_ind = np.where(profile[peak_ind:] < self.noise_param.saturation)[0] + peak_ind
            array[saturated_profile, alt_ind] = ma.masked
        return array

    def _find_saturated_profiles(self) -> np.ndarray:
        """Estimates saturated profiles using the variance of the top range gates."""
        var = _calc_var_from_top_gates(self.data['beta_raw'][:])
        return var < self.noise_param.variance

    @staticmethod
    def _remove_noise(array: np.ndarray,
                      noise: np.ndarray,
                      keep_negative: bool,
                      snr_limit: float) -> ma.MaskedArray:
        snr = array / utils.transpose(noise)
        if ma.isMaskedArray(array) is False:
            array = ma.masked_array(array)
        if keep_negative is True:
            array[np.abs(snr) < snr_limit] = ma.masked
        else:
            array[snr < snr_limit] = ma.masked
        return array

    def _calc_range_uncorrected(self, array: np.ndarray) -> np.ndarray:
        return array / self._get_range_squared()

    def _calc_range_corrected(self, array: np.ndarray) -> np.ndarray:
        return array * self._get_range_squared()

    def _get_range_squared(self) -> np.ndarray:
        """Returns range (m), squared and converted to km."""
        m2km = 0.001
        return (self.data['range'] * m2km) ** 2


def calc_sigma_units(time_vector: np.array, range_los: np.array) -> Tuple[float, float]:
    """Calculates Gaussian peak std parameters.

    The amount of smoothing is hard coded. This function calculates
    how many steps in time and height corresponds to this smoothing.

    Args:
        time_vector: 1D vector (fraction hour).
        range_los: 1D vector (m).

    Returns:
        tuple: Two element tuple containing number of steps in time and height to achieve wanted
            smoothing.

    """
    if len(time_vector) == 0 or np.max(time_vector) > 24:
        raise ValueError('Invalid time vector')
    minutes_in_hour = 60
    sigma_minutes = 2
    sigma_metres = 5
    time_step = utils.mdiff(time_vector) * minutes_in_hour
    alt_step = utils.mdiff(range_los)
    x_std = sigma_minutes / time_step
    y_std = sigma_metres / alt_step
    return x_std, y_std


def _estimate_clouds_from_beta(beta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Naively finds strong clouds from ceilometer backscatter."""
    cloud_limit = 1e-6
    cloud_ind = np.where(beta > cloud_limit)
    return cloud_ind, beta[cloud_ind], cloud_limit


def _estimate_background_noise(data: np.ndarray,
                               noise_min: float) -> np.ndarray:
    var = _calc_var_from_top_gates(data)
    noise = np.sqrt(var)
    noise[noise < noise_min] = noise_min
    return noise


def _calc_var_from_top_gates(data: np.ndarray) -> np.ndarray:
    fraction = 0.1
    n_gates = round(data.shape[1] * fraction)
    return ma.var(data[:, -n_gates:], axis=1)
