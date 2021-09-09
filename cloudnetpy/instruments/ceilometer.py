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

    def calc_beta(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts range-corrected raw beta to noise-screened beta."""
        is_saturation = self._find_saturated_profiles()
        beta = self._screen_data(self.processed_data['backscatter'], is_saturation, False)
        beta_smooth = ma.copy(self.processed_data['backscatter'])
        cloud_ind, cloud_values, cloud_limit = _estimate_clouds_from_beta(beta)
        beta_smooth[cloud_ind] = cloud_limit
        sigma = _calc_sigma_units(self.time, self.range)
        beta_smooth = scipy.ndimage.filters.gaussian_filter(beta_smooth, sigma)
        beta_smooth[cloud_ind] = cloud_values
        beta_smooth = self._screen_data(beta_smooth, is_saturation, True)
        return self.processed_data['backscatter'], beta, beta_smooth

    def calc_depol(self) -> Tuple[np.ndarray, np.ndarray]:
        """Converts raw depolarisation to noise-screened depolarisation."""
        snr_limit = 3
        is_saturation = self._find_saturated_profiles()
        x_pol = self._screen_data(self.processed_data['x_pol'], is_saturation, False, snr_limit)
        depol = x_pol / self.processed_data['p_pol']
        sigma = _calc_sigma_units(self.time, self.range)
        p_pol_smooth = ma.copy(self.processed_data['p_pol'])
        x_pol_smooth = ma.copy(self.processed_data['x_pol'])
        p_pol_smooth = scipy.ndimage.filters.gaussian_filter(p_pol_smooth, sigma)
        x_pol_smooth = scipy.ndimage.filters.gaussian_filter(x_pol_smooth, sigma)
        x_pol_smooth = self._screen_data(x_pol_smooth, is_saturation, True, snr_limit)
        depol_smooth = x_pol_smooth / p_pol_smooth
        return depol, depol_smooth

    def _screen_data(self,
                     array: np.ndarray,
                     is_saturation: np.ndarray,
                     is_smoothed: bool,
                     snr_limit: float = 5) -> np.ndarray:
        array = self._calc_range_uncorrected(array)
        array = self._screen_by_snr(array, snr_limit, is_saturation, is_smoothed)
        array = self._calc_range_corrected(array)
        return array

    def _screen_by_snr(self,
                       array: np.ndarray,
                       snr_limit: float,
                       is_saturation: np.ndarray,
                       is_smoothed: Optional[bool] = False) -> np.ndarray:
        """Screens noise from range-uncorrected lidar variable."""
        n_gates, _, saturation_noise, noise_min = self.noise_params
        noise_min = noise_min[0] if is_smoothed is True else noise_min[1]
        noise = _estimate_noise_from_top_gates(array, n_gates, noise_min)
        array = _reset_low_values_above_saturation(array, is_saturation, saturation_noise)
        array = _remove_noise(array, noise, snr_limit)
        return array

    def _find_saturated_profiles(self) -> np.ndarray:
        """Estimates saturated profiles using the variance of the top range gates."""
        n_gates, var_lim, _, _ = self.noise_params
        var = np.var(self.processed_data['backscatter'][:, -n_gates:], axis=1)
        return var < var_lim

    def _get_range_squared(self) -> np.ndarray:
        """Returns range (m), squared and converted to km."""
        m2km = 0.001
        return (self.range*m2km)**2

    def _calc_range_uncorrected(self, array: np.ndarray) -> np.ndarray:
        return array / self.range_squared

    def _calc_range_corrected(self, array: np.ndarray) -> np.ndarray:
        return array * self.range_squared


def _remove_noise(array: np.ndarray, noise: np.ndarray, snr_limit: float) -> np.ndarray:
    snr = array / utils.transpose(noise)
    array[np.abs(snr) < snr_limit] = ma.masked
    return array


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


def _reset_low_values_above_saturation(array: np.ndarray,
                                       is_saturation: np.ndarray,
                                       saturation_noise: float) -> np.ndarray:
    """Removes low values in saturated profiles above peak."""
    for saturated_profile in np.where(is_saturation)[0]:
        profile = array[saturated_profile, :]
        peak_ind = np.argmax(profile)
        alt_ind = np.where(profile[peak_ind:] < saturation_noise)[0] + peak_ind
        array[saturated_profile, alt_ind] = ma.masked
    return array
