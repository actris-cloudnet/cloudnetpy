""" This module has functions for liquid layer detection.
"""
import numpy as np
import numpy.ma as ma
import scipy.signal
from cloudnetpy import utils
from cloudnetpy.constants import T0


def base_ind(dprof, p, dist, lim):
    """ Find base index of a peak in profile.

    Return the lowermost index of profile where 1st order differences
    below the peak exceed a threshold value.

    Args:
        dprof (ndarray): 1st discrete difference profile of 1D array.
                         Masked values should be 0, e.g.
                         dprof=np.diff(masked_prof).filled(0)
        p (int): Index of (possibly local) peak in the original profile.
        dist (int): Number of elements investigated below **p**.
                    If (**p**-**dist**)<0, search starts from index 0.
        lim (float): Parameter for base index. Values greater than 1.0
                   are valid. Values close to 1 most likely return the
                   point right below the maximum 1st order difference
                   (within **dist** points below **p**).
                   Values larger than one more likely
                   accept some other point that is lower.

    Returns:
        Base index, or None if can't find it.

    Examples:
        Consider a profile

        >>> x = np.array([0, 0.5, 1, -99, 4, 8, 5])

        that contains one bad, masked value

        >>> mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0, 0, 0])
            [0 0.5, 1.0, --, 4.0, 8.0, 5.0]

        The 1st order difference is now

        >>> dx = np.diff(mx).filled(0)
            [0.5 0.5,  0. ,  0. ,  4. , -3. ]

        From the original profile we see that the peak index is 5.
        Let's assume our base can't be more than 4 elements below
        peak and the threshold value is 2. Thus we call

        >>> base_ind(dx, 5, 4, 2)
            4

        When x[4] is the lowermost point that satisfies the condition.
        Changing the threshold value would alter the result

        >>> base_ind(dx, 5, 4, 10)
            1
    """
    start = max(p-dist, 0)  # should not be negative
    diffs = dprof[start:p]
    mind = np.argmax(diffs)
    return start + np.where(diffs > diffs[mind]/lim)[0][0]


def top_ind(dprof, p, nprof, dist, lim):
    """ Find top index above peak."""
    end = min(p+dist, nprof)  # should not be greater than len(profile)
    diffs = dprof[p:end]
    mind = np.argmin(diffs)
    return p + np.where(diffs < diffs[mind]/lim)[0][-1] + 1


def find_liquid(beta, height, peak_amp=2e-5, max_width=300,
                min_points=3, min_top_der=2e-7):
    """ Estimate liquid layers from SNR-screened attenuated backscattering.

    Args:
        beta (MaskedArray): 2D attenuated backscattering.
        height (ndarray): 1D array of altitudes (m).
        peak_amp (float, optional): Minimum value for peak. Default is 2e-5.
        max_width (float, optional): Maximum width of peak. Default is 300 (m).
        min_points (int, optional): Minimum number of valid points in peak.
            Default is 3.
        min_top_der (float, optional): Minimum derivative above peak
            defined as (beta_peak-beta_top) / (alt_top-alt_peak) which
            is always positive. Default is 2e-7.

    Returns:
        Boolean array denoting the liquid layers.

    """
    is_liquid, liquid_top, liquid_base = utils.init(3, beta.shape, dtype=bool,
                                                    masked=False)
    base_below_peak = utils.number_of_elements(height, 200)
    top_above_peak = utils.number_of_elements(height, 150)
    beta_diff = np.diff(beta, axis=1).filled(0)
    beta = beta.filled(0)
    pind = scipy.signal.argrelextrema(beta, np.greater, order=4, axis=1)
    strong_peaks = np.where(beta[pind] > peak_amp)
    pind = (pind[0][strong_peaks], pind[1][strong_peaks])
    for n, peak in zip(*pind):
        lprof = beta[n, :]
        dprof = beta_diff[n, :]
        try:
            base = base_ind(dprof, peak, base_below_peak, 4)
        except:
            continue
        try:
            top = top_ind(dprof, peak, height.shape[0], top_above_peak, 4)
        except:
            continue
        npoints = np.count_nonzero(lprof[base:top+1])
        peak_width = height[top] - height[base]
        top_der = (lprof[peak] - lprof[top]) / (height[top] - height[peak])
        conds = (npoints > min_points,
                 peak_width < max_width,
                 top_der > min_top_der)
        if all(conds):
            is_liquid[n, base:top+1] = True
            liquid_top[n, top] = True
            liquid_base[n, base] = True
    return is_liquid, liquid_base, liquid_top


def correct_liquid_top(Z, Tw, is_freezing, is_liquid, liquid_top, height):
    """Corrects lidar detected liquid cloud top using radar data.

    Args:
        Z (MaskedArray): Radar echo.
        Tw (ndarray): Wet bulb temperature.
        is_freezing (ndarray): Boolean field of sub-zero temperature
            that was fixed using the melting layer.
        is_liquid (ndarray): Boolean field of liquid cloud droplets.
        liquid_top (ndarray): Boolean field of liquid cloud tops.
        height (ndarray): Altitude vector.

    Returns:
        Corrected liquid cloud field.

    """
    top_above = utils.number_of_elements(height, 750)
    for prof, top in zip(*np.where(liquid_top)):
        ind = np.where(is_freezing[prof, top:])[0][0] + top_above
        rad = Z[prof, top:top+ind+1]
        if not (rad.mask.all() or ~rad.mask.any()):
            first_masked = ma.where(rad.mask)[0][0]
            is_liquid[prof, top:top+first_masked+1] = True
    is_liquid[Tw < (T0-40)] = False
    return is_liquid
