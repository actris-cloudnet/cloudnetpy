"""This module has functions for liquid layer detection."""

import numpy as np
import scipy.signal
from numpy import ma

from cloudnetpy import utils
from cloudnetpy.categorize import atmos_utils
from cloudnetpy.categorize.containers import ClassData


def correct_liquid_top(
    obs: ClassData,
    is_liquid: np.ndarray,
    is_freezing: np.ndarray,
    limit: float = 200,
) -> np.ndarray:
    """Corrects lidar detected liquid cloud top using radar data.

    Args:
        obs: The :class:`ClassData` instance.
        is_liquid: 2-D boolean array denoting liquid clouds from lidar data.
        is_freezing: 2-D boolean array of sub-zero temperature, derived from the model
            temperature and melting layer based on radar data.
        limit: The maximum correction distance (m) above liquid cloud top.

    Returns:
        Corrected liquid cloud array.

    References:
        Hogan R. and O'Connor E., 2004, https://bit.ly/2Yjz9DZ.

    """
    is_liquid_corrected = np.copy(is_liquid)
    liquid_tops = atmos_utils.find_cloud_tops(is_liquid)
    top_above = utils.n_elements(obs.height, limit)
    for prof, top in zip(*np.where(liquid_tops), strict=True):
        ind = _find_ind_above_top(is_freezing[prof, top:], top_above)
        rad = obs.z[prof, top : top + ind + 1]
        if not (rad.mask.all() or ~rad.mask.any()):
            first_masked = ma.where(rad.mask)[0][0]
            is_liquid_corrected[prof, top : top + first_masked] = True
    return is_liquid_corrected


def _find_ind_above_top(is_freezing_from_peak: np.ndarray, top_above: int) -> int:
    first_point_below_zero = np.where(is_freezing_from_peak)[0][0]
    ind = first_point_below_zero + top_above
    return min(len(is_freezing_from_peak) - 1, ind)


def find_liquid(
    obs: ClassData,
    peak_amp: float = 1e-6,
    max_width: float = 300,
    min_points: int = 3,
    min_top_der: float = 1e-7,
    min_lwp: float = 0,
    min_alt: float = 100,
) -> np.ndarray:
    """Estimate liquid layers from SNR-screened attenuated backscatter.

    Args:
        obs: The :class:`ClassData` instance.
        peak_amp: Minimum value of peak. Default is 1e-6.
        max_width: Maximum width of peak. Default is 300 (m).
        min_points: Minimum number of valid points in peak. Default is 3.
        min_top_der: Minimum derivative above peak, defined as
            (beta_peak-beta_top) / (alt_top-alt_peak). Default is 1e-7.
        min_lwp: Minimum value from linearly interpolated lwp (kg m-2)
            measured by the mwr. Default is 0.
        min_alt: Minimum altitude of the peak from the ground. Default is 100 (m).

    Returns:
        2-D boolean array denoting liquid layers.

    References:
        The method is based on Tuononen, M. et.al, 2019,
        https://acp.copernicus.org/articles/19/1985/2019/.

    """

    def _is_proper_peak() -> bool:
        conditions = (
            npoints >= min_points,
            peak_width < max_width,
            top_der > min_top_der,
            is_positive_lwp,
            peak_alt > min_alt,
        )
        return all(conditions)

    lwp_int = interpolate_lwp(obs)
    beta = ma.copy(obs.beta)
    height = obs.height

    is_liquid = np.zeros(beta.shape, dtype=bool)
    base_below_peak = utils.n_elements(height, 200)
    top_above_peak = utils.n_elements(height, 150)
    difference = ma.array(np.diff(beta, axis=1))
    beta_diff = difference.filled(0)
    beta = beta.filled(0)
    peak_indices = _find_strong_peaks(beta, peak_amp)

    for n, peak in zip(*peak_indices, strict=True):
        lprof = beta[n, :]
        dprof = beta_diff[n, :]
        try:
            base = ind_base(dprof, peak, base_below_peak, 4)
            top = ind_top(dprof, peak, height.shape[0], top_above_peak, 4)
        except IndexError:
            continue
        npoints = np.count_nonzero(lprof[base : top + 1])
        peak_width = height[top] - height[base]
        peak_alt = height[peak] - height[0]
        top_der = (lprof[peak] - lprof[top]) / (height[top] - height[peak])
        is_positive_lwp = lwp_int[n] >= min_lwp
        if _is_proper_peak():
            is_liquid[n, base : top + 1] = True

    return is_liquid


def ind_base(dprof: np.ndarray, ind_peak: int, dist: int, lim: float) -> int:
    """Finds base index of a peak in profile.

    Return the lowermost index of profile where 1st order differences
    below the peak exceed a threshold value.

    Args:
        dprof: 1-D array of 1st discrete difference. Masked values should
            be 0, e.g. dprof = np.diff(masked_prof).filled(0)
        ind_peak: Index of (possibly local) peak in the original profile.
            Note that the peak must be found with some other method before
            calling this function.
        dist: Number of elements investigated below *p*. If ( *p* - *dist*)<0,
            search starts from index 0.
        lim: Parameter for base index. Values greater than 1.0 are valid.
            Values close to 1 most likely return the point right below the
            maximum 1st order difference (within *dist* points below *p*).
            Values larger than 1 more likely accept some other point, lower
            in the profile.

    Returns:
        Base index of the peak.

    Raises:
        IndexError: Can't find proper base index (probably too many masked
            values in the profile).

    Examples:
        Consider a profile

        >>> x = np.array([0, 0.5, 1, -99, 4, 8, 5])

        that contains one bad, masked value

        >>> mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0, 0, 0])
            [0, 0.5, 1.0, --, 4.0, 8.0, 5.0]

        The 1st order difference is now

        >>> dx = np.diff(mx).filled(0)
            [0.5, 0.5, 0, 0, 4, -3]

        From the original profile we see that the peak index is 5.
        Let's assume our base can't be more than 4 elements below
        peak and the threshold value is 2. Thus we call

        >>> ind_base(dx, 5, 4, 2)
            4

        When x[4] is the lowermost point that satisfies the condition.
        Changing the threshold value would alter the result

        >>> ind_base(dx, 5, 4, 10)
            1

    See Also:
        droplet.ind_top()

    """
    start = max(ind_peak - dist, 0)  # should not be negative
    diffs = dprof[start:ind_peak]
    mind = np.argmax(diffs)
    return start + np.where(diffs > diffs[mind] / lim)[0][0]


def ind_top(dprof: np.ndarray, ind_peak: int, nprof: int, dist: int, lim: float) -> int:
    """Finds top index of a peak in profile.

    Return the uppermost index of profile where 1st order differences
    above the peak exceed a threshold value.

    Args:
        dprof: 1-D array of 1st discrete difference. Masked values should be 0, e.g.
            dprof = np.diff(masked_prof).filled(0)
        nprof: Length of the profile. Top index can't be higher than this.
        ind_peak: Index of (possibly local) peak in the profile. Note that the peak
            must be found with some other method before calling this function.
        dist: Number of elements investigated above *p*. If (*p* + *dist*) > *nprof*,
            search ends to *nprof*.
        lim: Parameter for top index. Values greater than 1.0 are valid. Values close
            to 1 most likely return the point right above the maximum 1st order
            difference (within *dist* points above *p*). Values larger than 1 more
            likely accept some other point, higher in the profile.

    Returns:
        Top index of the peak.

    Raises:
        IndexError: Can not find proper top index (probably too many masked
            values in the profile).

    See Also:
        droplet.ind_base()

    """
    end = min(ind_peak + dist, nprof)  # should not be greater than len(profile)
    diffs = dprof[ind_peak:end]
    mind = np.argmin(diffs)
    return ind_peak + np.where(diffs < diffs[mind] / lim)[0][-1] + 1


def interpolate_lwp(obs: ClassData) -> np.ndarray:
    """Linear interpolation of liquid water path to fill masked values.

    Args:
        obs: The :class:`ClassData` instance.

    Returns:
        Liquid water path where the masked values are filled by interpolation.

    """
    if obs.lwp.all() is ma.masked:
        return np.zeros(obs.time.shape)
    ind = ma.where(obs.lwp)
    return np.interp(obs.time, obs.time[ind], obs.lwp[ind])


def _find_strong_peaks(data: np.ndarray, threshold: float) -> tuple:
    """Finds local maximums from data (greater than *threshold*)."""
    peaks = scipy.signal.argrelextrema(data, np.greater, order=4, axis=1)
    strong_peaks = np.where(data[peaks] > threshold)
    return peaks[0][strong_peaks], peaks[1][strong_peaks]
