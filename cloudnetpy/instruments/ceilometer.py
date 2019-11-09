import numpy as np
import numpy.ma as ma
import scipy.ndimage
from cloudnetpy import utils


class Ceilometer:
    """Base class for all types of ceilometers."""

    def __init__(self, file_name):
        self.file_name = file_name
        self.model = ''
        self.backscatter = np.array([])
        self.data = {}
        self.metadata = {}
        self.range = np.array([])
        self.time = []
        self.date = []
        self.noise_params = (1, 1, 1, (1, 1))

    def calc_beta(self):
        """Converts range-corrected raw beta to noise-screened beta."""

        def _screen_beta(beta_in, smooth):
            beta_in = _calc_range_uncorrected_beta(beta_in, range_squared)
            beta_in = self._screen_by_snr(beta_in, is_saturation, smooth=smooth)
            return _calc_range_corrected_beta(beta_in, range_squared)

        range_squared = _get_range_squared(self.range)
        is_saturation = self._find_saturated_profiles()
        beta = _screen_beta(self.backscatter, False)
        # smoothed version:
        beta_smooth = ma.copy(self.backscatter)
        cloud_ind, cloud_values, cloud_limit = _estimate_clouds_from_beta(beta)
        beta_smooth[cloud_ind] = cloud_limit
        sigma = _calc_sigma_units(self.time, self.range)
        beta_smooth = scipy.ndimage.filters.gaussian_filter(beta_smooth, sigma)
        beta_smooth[cloud_ind] = cloud_values
        beta_smooth = _screen_beta(beta_smooth, True)
        return self.backscatter, beta, beta_smooth

    def _screen_by_snr(self, beta_uncorrected, is_saturation, smooth=False):
        """Screens noise from ceilometer backscatter.

        Args:
            beta_uncorrected (ndarray): Range-uncorrected backscatter.
            is_saturation (ndarray): Boolean array denoting saturated profiles.
            smooth (bool, optional): Should be true if input beta is smoothed.
                Default is False.

        """
        beta = ma.copy(beta_uncorrected)
        n_gates, _, saturation_noise, noise_min = self.noise_params
        noise_min = noise_min[0] if smooth else noise_min[1]
        noise = _estimate_noise_from_top_gates(beta, n_gates, noise_min)
        beta = _reset_low_values_above_saturation(beta, is_saturation, saturation_noise)
        beta = _remove_noise(beta, noise)
        return beta

    def _find_saturated_profiles(self):
        """Estimates saturated profiles using the variance of the top range gates."""
        n_gates, var_lim, _, _ = self.noise_params
        var = np.var(self.backscatter[:, -n_gates:], axis=1)
        return var < var_lim


def _remove_noise(beta_in, noise):
    beta = ma.copy(beta_in)
    snr_limit = 5
    snr = (beta.T / noise)
    beta[snr.T < snr_limit] = ma.masked
    return beta


def _calc_sigma_units(time, range_instru):
    """Calculates Gaussian peak std parameters.

    The amount of smoothing is hard coded. This function calculates
    how many steps in time and height corresponds to this smoothing.

    Args:
        time (ndarray): 1D vector (fraction hour).
        range_instru (ndarray): 1D vector (m).

    Returns:
        tuple: Two element tuple containing number of steps in time and height
            to achieve wanted smoothing.

    """
    minutes_in_hour = 60
    sigma_minutes = 2
    sigma_metres = 5
    time_step = utils.mdiff(time) * minutes_in_hour
    alt_step = utils.mdiff(range_instru)
    x_std = sigma_minutes / time_step
    y_std = sigma_metres / alt_step
    return x_std, y_std


def _estimate_clouds_from_beta(beta):
    """Naively finds strong clouds from ceilometer backscatter."""
    cloud_limit = 1e-6
    cloud_ind = np.where(beta > cloud_limit)
    return cloud_ind, beta[cloud_ind], cloud_limit


def _estimate_noise_from_top_gates(beta, n_gates, noise_min):
    """Estimates backscatter noise from topmost range gates."""
    noise = ma.std(beta[:, -n_gates:], axis=1)
    noise[noise < noise_min] = noise_min
    return noise


def _reset_low_values_above_saturation(beta_in, is_saturation, saturation_noise):
    """Removes low values in saturated profiles above peak."""
    beta = ma.copy(beta_in)
    for saturated_profile in np.where(is_saturation)[0]:
        profile = beta[saturated_profile, :]
        peak_ind = np.argmax(profile)
        alt_ind = np.where(profile[peak_ind:] < saturation_noise)[0] + peak_ind
        beta[saturated_profile, alt_ind] = ma.masked
    return beta


def _calc_range_uncorrected_beta(beta, range_squared):
    """Calculates range uncorrected beta.

    Args:
        beta (ndarray): 2D attenuated backscatter.
        range_squared (ndarray): 1D altitude vector (km), squared.

    Returns:
        ndarray: 2D range uncorrected beta.

    """
    return beta / range_squared


def _calc_range_corrected_beta(beta, range_squared):
    """Calculates range corrected beta.

    Args:
        beta (ndarray): 2D attenuated backscatter.
        range_squared (ndarray): 1D altitude vector (km), squared.

    Returns:
        ndarray: 2D range corrected beta.

    """
    return beta * range_squared


def _get_range_squared(range_instru):
    """Returns range (m), squared and converted to km."""
    m2km = 0.001
    return (range_instru*m2km)**2
