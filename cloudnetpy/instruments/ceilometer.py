from typing import Tuple, Optional
import numpy as np
import numpy.ma as ma
import scipy.ndimage
from cloudnetpy import utils


class Ceilometer:
    """Base class for all types of ceilometers."""

    def __init__(self, full_path: str):
        self.file_name = full_path
        self.model = ''
        self.processed_data = {}
        self.data = {}
        self.metadata = {}
        self.range = np.array([])
        self.range_squared = np.array([])
        self.time = []
        self.date = []
        self.noise_params = (1, 1, 1, (1, 1))
        self.calibration_factor = 1

    def calc_beta(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts range-corrected raw beta to noise-screened beta."""
        snr_limit = 5
        noisy_data = NoisyData(*self._get_args(), snr_limit)
        beta = noisy_data.screen_data(self.processed_data['backscatter'])
        beta_smooth = self._calc_beta_smooth(beta)
        beta_smooth = noisy_data.screen_data(beta_smooth, is_smoothed=True)
        return self.processed_data['backscatter'], beta, beta_smooth

    def calc_depol(self) -> Tuple[np.ndarray, np.ndarray]:
        """Converts raw depolarisation to noise-screened depolarisation."""
        snr_limit = 3
        noisy_data = NoisyData(*self._get_args(), snr_limit)
        x_pol = noisy_data.screen_data(self.processed_data['x_pol'], keep_negative=True)
        depol = x_pol / self.processed_data['p_pol']
        sigma = _calc_sigma_units(self.time, self.range)
        p_pol_smooth = scipy.ndimage.filters.gaussian_filter(self.processed_data['p_pol'], sigma)
        x_pol_smooth = scipy.ndimage.filters.gaussian_filter(self.processed_data['x_pol'], sigma)
        x_pol_smooth = noisy_data.screen_data(x_pol_smooth, is_smoothed=True)
        depol_smooth = x_pol_smooth / p_pol_smooth
        return depol, depol_smooth

    def _calc_beta_smooth(self, beta: np.ndarray) -> np.ndarray:
        beta_smooth = ma.copy(self.processed_data['backscatter'])
        cloud_ind, cloud_values, cloud_limit = _estimate_clouds_from_beta(beta)
        beta_smooth[cloud_ind] = cloud_limit
        sigma = _calc_sigma_units(self.time, self.range)
        beta_smooth = scipy.ndimage.filters.gaussian_filter(beta_smooth, sigma)
        beta_smooth[cloud_ind] = cloud_values
        return beta_smooth

    def _get_range_squared(self) -> np.ndarray:
        """Returns range (m), squared and converted to km."""
        m2km = 0.001
        return (self.range * m2km) ** 2

    def _get_args(self):
        return self.processed_data, self.range_squared, self.noise_params


class NoisyData:
    def __init__(self,
                 data: dict,
                 range_squared: np.ndarray,
                 noise_params: tuple,
                 snr_limit: float):
        self.data = data
        self.range_squared = range_squared
        self.noise_params = noise_params
        self.snr_limit = snr_limit
        self._is_saturation = self._find_saturated_profiles()

    def screen_data(self,
                    array: np.ndarray,
                    is_smoothed: Optional[bool] = False,
                    keep_negative: Optional[bool] = False) -> np.ndarray:
        array = self._calc_range_uncorrected(array)
        array = self._screen_by_snr(array, is_smoothed, keep_negative)
        array = self._calc_range_corrected(array)
        return array

    def _screen_by_snr(self,
                       array: np.ndarray,
                       is_smoothed: bool,
                       keep_negative: bool) -> np.ndarray:
        """Screens noise from range-uncorrected lidar variable."""
        n_gates, _, saturation_noise, noise_min = self.noise_params
        noise_min = noise_min[0] if is_smoothed is True else noise_min[1]
        noise = _estimate_noise_from_top_gates(array, n_gates, noise_min)
        array = self._reset_low_values_above_saturation(array, saturation_noise)
        array = self._remove_noise(array, noise, keep_negative)
        return array

    def _find_saturated_profiles(self) -> np.ndarray:
        """Estimates saturated profiles using the variance of the top range gates."""
        n_gates, var_lim, _, _ = self.noise_params
        var = np.var(self.data['backscatter'][:, -n_gates:], axis=1)
        return var < var_lim

    def _reset_low_values_above_saturation(self,
                                           array: np.ndarray,
                                           saturation_noise: float) -> np.ndarray:
        """Removes low values in saturated profiles above peak."""
        for saturated_profile in np.where(self._is_saturation)[0]:
            profile = array[saturated_profile, :]
            peak_ind = np.argmax(profile)
            alt_ind = np.where(profile[peak_ind:] < saturation_noise)[0] + peak_ind
            array[saturated_profile, alt_ind] = ma.masked
        return array

    def _remove_noise(self,
                      array: np.ndarray,
                      noise: np.ndarray,
                      keep_negative: bool) -> np.ndarray:
        snr = array / utils.transpose(noise)
        if keep_negative is True:
            array[np.abs(snr) < self.snr_limit] = ma.masked
        else:
            array[snr < self.snr_limit] = ma.masked
        return array

    def _calc_range_uncorrected(self, array: np.ndarray) -> np.ndarray:
        return array / self.range_squared

    def _calc_range_corrected(self, array: np.ndarray) -> np.ndarray:
        return array * self.range_squared


def _calc_sigma_units(time_vector: list, range_los: np.ndarray) -> Tuple[float, float]:
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


def _estimate_noise_from_top_gates(data: np.ndarray, n_gates: int, noise_min: float) -> np.ndarray:
    """Estimates noise from topmost range gates."""
    noise = ma.std(data[:, -n_gates:], axis=1)
    noise[noise < noise_min] = noise_min
    return noise
