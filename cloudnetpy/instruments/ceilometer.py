from typing import Tuple, Optional
import numpy as np
import numpy.ma as ma
import scipy.ndimage
from cloudnetpy import utils
from cloudnetpy.cloudnetarray import CloudnetArray
import logging


class NoiseParam:
    """Noise parameters. Values are weakly instrument-dependent."""
    def __init__(self,
                 noise_min: Optional[float] = 1e-9,
                 noise_smooth_min: Optional[float] = 4e-9):
        self.noise_min = noise_min
        self.noise_smooth_min = noise_smooth_min


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
                              snr_limit: Optional[int] = 5,
                              range_corrected: Optional[bool] = True) -> np.ndarray:
        """Screens noise from lidar variable."""
        noisy_data = NoisyData(self.data, self.noise_param, range_corrected)
        array_screened = noisy_data.screen_data(array, snr_limit=snr_limit)
        return array_screened

    def calc_beta_smooth(self,
                         beta: np.ndarray,
                         snr_limit: Optional[int] = 5,
                         range_corrected: Optional[bool] = True) -> np.ndarray:
        noisy_data = NoisyData(self.data, self.noise_param, range_corrected)
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
    def __init__(self, data: dict, noise_param: NoiseParam, range_corrected: bool):
        self.data = data
        self.noise_param = noise_param
        self.range_corrected = range_corrected
        self._signal = None

    def screen_data(self,
                    data: np.array,
                    snr_limit: Optional[float] = 5,
                    is_smoothed: Optional[bool] = False,
                    keep_negative: Optional[bool] = False) -> np.ndarray:
        filter_fog = True
        filter_negatives = True
        filter_snr = True
        original_data = ma.copy(data)
        self._signal = original_data
        range_uncorrected = self._calc_range_uncorrected(original_data)
        noise_min = self.noise_param.noise_smooth_min if is_smoothed is True else self.noise_param.noise_min
        noise = _estimate_background_noise(range_uncorrected)
        noise_below_threshold = noise < noise_min
        logging.info(f'Adjusted noise of {sum(noise_below_threshold)} profiles')
        noise[noise_below_threshold] = noise_min
        if filter_negatives:
            is_negative = self._remove_low_values_above_consequent_negatives(range_uncorrected)
            noise[is_negative] = 1e-12
        if filter_fog:
            is_fog = self._find_fog_profiles()
            self._clean_fog_profiles(range_uncorrected, is_fog)
            noise[is_fog] = 1e-12
        if filter_snr:
            range_uncorrected = self._remove_noise(range_uncorrected, noise, keep_negative, snr_limit)
        range_corrected = self._calc_range_corrected(range_uncorrected)
        return range_corrected

    @staticmethod
    def _remove_low_values_above_consequent_negatives(range_uncorrected: np.ndarray) -> np.ndarray:
        n_negatives = 5
        n_gates = 100
        threshold = 3e-6
        negative_data = range_uncorrected[:, :n_gates] < 0
        n_consequent_negatives = utils.cumsumr(negative_data, axis=1)
        time_ind, alt_ind = np.where(n_consequent_negatives > n_negatives)
        for time, alt in zip(time_ind, alt_ind):
            profile = range_uncorrected[time, alt:]
            profile[profile < threshold] = ma.masked
            range_uncorrected[time, alt:] = profile
        cleaned_indices = np.unique(time_ind)
        logging.info(f'Cleaned {len(cleaned_indices)} profiles with negative filter')
        return np.unique(time_ind)

    def _find_fog_profiles(self) -> np.ndarray:
        """Finds saturated profiles from beta_raw (can be range-corrected or range-uncorrected)."""
        n_gates = 20
        signal_sum_threshold = 1e-3
        variance_threshold = 1e-15
        signal_sum = ma.sum(ma.abs(self.data['beta_raw'][:, :n_gates]), axis=1)
        var = _calc_var_from_top_gates(self.data['beta_raw'])
        is_fog = (signal_sum > signal_sum_threshold) | (var < variance_threshold)
        logging.info(f'Cleaned {sum(is_fog)} profiles with fog filter')
        return is_fog

    def _remove_noise(self,
                      array: np.ndarray,
                      noise: np.ndarray,
                      keep_negative: bool,
                      snr_limit: float) -> ma.MaskedArray:
        snr = array / utils.transpose(noise)
        if self.range_corrected is False:
            snr_scale_factor = 6
            ind = self._get_altitude_ind()
            snr[:, ind] *= snr_scale_factor
        if ma.isMaskedArray(array) is False:
            array = ma.masked_array(array)
        if keep_negative is True:
            array[np.abs(snr) < snr_limit] = ma.masked
        else:
            array[snr < snr_limit] = ma.masked
        return array

    def _calc_range_uncorrected(self, array: np.ndarray) -> np.ndarray:
        ind = self._get_altitude_ind()
        array[:, ind] = array[:, ind] / self._get_range_squared()[ind]
        return array

    def _calc_range_corrected(self, array: np.ndarray) -> np.ndarray:
        ind = self._get_altitude_ind()
        array[:, ind] = array[:, ind] * self._get_range_squared()[ind]
        return array

    def _get_altitude_ind(self):
        alt_limit = 2400
        if self.range_corrected is False:
            logging.warning(f'Raw data not range-corrected, correcting below {alt_limit} m')
            return np.where(self.data['range'] < alt_limit)
        else:
            return np.arange(len(self.data['range']))

    def _get_range_squared(self) -> np.ndarray:
        """Returns range (m), squared and converted to km."""
        m2km = 0.001
        return (self.data['range'] * m2km) ** 2

    def _clean_fog_profiles(self, array: np.ndarray, is_fog: np.ndarray) -> None:
        """Removes values in saturated (e.g. fog) profiles above peak."""
        threshold = 2e-6
        for time_ind in np.where(is_fog)[0]:
            signal = self._signal[time_ind, :]
            peak_ind = np.argmax(signal)
            array[time_ind, peak_ind:][signal[peak_ind:] < threshold] = ma.masked


def _estimate_background_noise(data: np.ndarray) -> np.ndarray:
    var = _calc_var_from_top_gates(data)
    return np.sqrt(var)


def _calc_var_from_top_gates(data: np.ndarray) -> np.ndarray:
    fraction = 0.1
    n_gates = round(data.shape[1] * fraction)
    return ma.var(data[:, -n_gates:], axis=1)


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
