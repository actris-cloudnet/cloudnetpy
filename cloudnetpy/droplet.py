""" This module has functions for liquid layer detection.
"""
import numpy as np
import numpy.ma as ma
import scipy.signal
from cloudnetpy import utils


def find_liquid(obs, peak_amp=1e-6,
                max_width=300,
                min_points=3,
                min_top_der=1e-7,
                min_lwp=0):
    """ Estimate liquid layers from SNR-screened attenuated backscattering.

    Args:
        obs (ClassData): Observations container.
        peak_amp (float, optional): Minimum value of peak. Default is 2e-5.
        max_width (float, optional): Maximum width of peak. Default is 300 (m).
        min_points (int, optional): Minimum number of valid points in peak.
            Default is 3.
        min_top_der (float, optional): Minimum derivative above peak,
            defined as (beta_peak-beta_top) / (alt_top-alt_peak), which
            is always positive. Default is 2e-7.
        min_lwp (float, optional): Minimum value from linearly interpolated lwp
            measured by the mwr. Default is 0.

    Returns:
        dict: Dict containing 'presence', 'bases' and 'tops'.

    """
    def _find_strong_peaks():
        peaks = scipy.signal.argrelextrema(beta, np.greater, order=4, axis=1)
        strong_peaks = np.where(beta[peaks] > peak_amp)
        return peaks[0][strong_peaks], peaks[1][strong_peaks]

    def _is_proper_peak():
        conditions = (npoints > min_points,
                      peak_width < max_width,
                      top_der > min_top_der,
                      is_positive_lwp)
        return all(conditions)

    def _save_peak_position():
        is_liquid[n, base:top + 1] = True
        liquid_top[n, top] = True
        liquid_base[n, base] = True

    def _interpolate_lwp():
        ind = ma.where(obs.lwp)
        return np.interp(obs.time, obs.time[ind], obs.lwp[ind])

    lwp_int = _interpolate_lwp()
    beta = obs.beta

    # TODO: append zero-row into data instead of setting first values to zero.
    # This fix is because the peak can be the very first value (thus there is no proper base in data)
    beta[:, 0] = 0
    height = obs.height

    is_liquid, liquid_top, liquid_base = utils.init(3, beta.shape, dtype=bool,
                                                    masked=False)
    base_below_peak = utils.n_elements(height, 200)
    top_above_peak = utils.n_elements(height, 150)
    beta_diff = np.diff(beta, axis=1).filled(0)
    beta = beta.filled(0)
    peak_indices = _find_strong_peaks()

    for n, peak in zip(*peak_indices):
        lprof = beta[n, :]
        dprof = beta_diff[n, :]
        try:
            base = ind_base(dprof, peak, base_below_peak, 4)
            top = ind_top(dprof, peak, height.shape[0], top_above_peak, 4)
        except:
            continue
        npoints = np.count_nonzero(lprof[base:top+1])
        peak_width = height[top] - height[base]
        top_der = (lprof[peak] - lprof[top]) / (height[top] - height[peak])
        is_positive_lwp = lwp_int[n] > min_lwp
        if _is_proper_peak():
            _save_peak_position()
    return {'presence': is_liquid,
            'bases': liquid_base,
            'tops': liquid_top}


def ind_base(dprof, p, dist, lim):
    """Finds base index of a peak in profile.

    Return the lowermost index of profile where 1st order differences
    below the peak exceed a threshold value.

    Args:
        dprof (ndarray): 1-D array of 1st discrete difference.
            Masked values should be 0, e.g. dprof =
            np.diff(masked_prof).filled(0)
        p (int): Index of (possibly local) peak in the original profile.
            Note that the peak must be found with some other method prior
            calling this function.
        dist (int): Number of elements investigated below *p*.
                    If ( *p* - *dist*)<0, search starts from index 0.
        lim (float): Parameter for base index. Values greater than 1.0
                   are valid. Values close to 1 most likely return the
                   point right below the maximum 1st order difference
                   (within *dist* points below *p*).
                   Values larger than 1 more likely
                   accept some other point, lower in the profile.

    Returns:
        int: Base index of the peak.

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

        >>> ind_base(dx, 5, 4, 2)
            4

        When x[4] is the lowermost point that satisfies the condition.
        Changing the threshold value would alter the result

        >>> ind_base(dx, 5, 4, 10)
            1

    See also:
        droplet.ind_top()

    """
    start = max(p-dist, 0)  # should not be negative
    diffs = dprof[start:p]
    mind = np.argmax(diffs)
    return start + np.where(diffs > diffs[mind]/lim)[0][0]


def ind_top(dprof, p, nprof, dist, lim):
    """Finds top index of a peak in profile.

    Return the uppermost index of profile where 1st order differences
    above the peak exceed a threshold value.

    Args:
        dprof (ndarray): 1-D array of 1st discrete difference.
            Masked values should be 0, e.g. dprof =
            np.diff(masked_prof).filled(0)
        nprof (int): Length of the profile. Top index can't be higher
            than this.
        p (int): Index of (possibly local) peak in the profile.
            Note that the peak must be found with some other method prior
            calling this function.
        dist (int): Number of elements investigated above *p*.
                    If (*p* + *dist*) > *nprof*, search ends to *nprof*.
        lim (float): Parameter for top index. Values greater than 1.0
                   are valid. Values close to 1 most likely return the
                   point right above the maximum 1st order difference
                   (within *dist* points above *p*).
                   Values larger than 1 more likely
                   accept some other point, higher in the profile.

    Returns:
        int: Top index of the peak.

    See also:
        droplet.ind_base()

    """
    end = min(p+dist, nprof)  # should not be greater than len(profile)
    diffs = dprof[p:end]
    mind = np.argmin(diffs)
    return p + np.where(diffs < diffs[mind]/lim)[0][-1] + 1


def correct_liquid_top(obs, liquid, is_freezing, limit=200):
    """Corrects lidar detected liquid cloud top using radar data.

    Args:
        obs (ClassData): Observations container.
        liquid (dict): Dictionary for liquid clouds.
        is_freezing (ndarray): 2-D boolean array of sub-zero temperature,
            derived from the model temperature and melting layer based
            on radar data.
        limit (float): The maximum correction distance (m) above liquid cloud top.
    Returns:
        ndarray: Corrected liquid cloud array.
    See also:
        droplet.find_liquid()
    """
    top_above = utils.n_elements(obs.height, limit)
    for prof, top in zip(*np.where(liquid['tops'])):
        ind = np.where(is_freezing[prof, top:])[0][0] + top_above
        rad = obs.z[prof, top:top+ind+1]
        if not (rad.mask.all() or ~rad.mask.any()):
            first_masked = ma.where(rad.mask)[0][0]
            liquid['presence'][prof, top:top+first_masked] = True
    return liquid['presence']
