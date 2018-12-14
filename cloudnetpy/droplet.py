""" This module has functions for liquid layer detection.
"""

import numpy as np
import numpy.ma as ma
import scipy.signal
import utils


def get_base_ind(dprof, p, dist, lim):
    """ Find base index of a peak in profile.

    Return the lowermost index of profile where 1st order differences
    below the peak exceed a threshold value.

    Args:
        dprof (array_like): 1st discrete difference profile of 1D array.
                            Masked values should be 0, e.g.
                            dprof=np.diff(masked_prof).filled(0)
        p (int): Index of (possibly local) peak in the original profile.
        dist (int): Number of elements investigated below **p**.
                    If (**p**-**dist**)<0, search starts from index 0.
        lim (float): Parameter for base index. Values greater than 1.0 are valid.
                   Values close to 1 most likely return the point right
                   below the maximum 1st order difference
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
        >>> [0 0.5, 1.0, --, 4.0, 8.0, 5.0]

        The 1st order difference is now

        >>> dx = np.diff(mx).filled(0)
        >>> [0.5 0.5,  0. ,  0. ,  4. , -3. ]

        From the original profile we see that the peak index is 5. Let's say our
        base can't be more than 4 elements below peak and the threshold value is 2.
        Thus we call

        >>> get_base_ind(dx, 5, 4, 2)
        >>> 4

        When x[4] is the lowermost point that satisfies the condition.
        Changing the threshold value would alter the result

        >>> get_base_ind(dx, 5, 4, 10)
        >>> 1
    """
    start = max(p-dist, 0)  # should not be negative
    diffs = dprof[start:p]
    mind = np.argmax(diffs)
    return start + np.where(diffs > diffs[mind]/lim)[0][0]


def get_top_ind(dprof, p, nprof, dist, lim):
    """ Find top index above peak."""
    end = min(p+dist, nprof)  # should not be greater than len(profile)
    diffs = dprof[p:end]
    mind = np.argmin(diffs)
    return p + np.where(diffs < diffs[mind]/lim)[0][-1] + 1


def get_liquid_layers(beta, height, peak_amp=2e-5, max_width=300,
                      min_points=3, min_top_der=4e-7):
    """ Estimate liquid layers from SNR-screened attenuated backscattering.

    Args:
        beta (array_like): 2D attenuated backscattering.
        height (array_like): 1D array of altitudes (m).
        peak_amp (float, optional): Minimum value for peak. Default is 2e-5.
        max_width (float, optional): Maximum width of peak. Default is 300 (m).
        min_points (int, optional): Minimum number of valid points in peak. Default is 3.
        min_top_der (float, optional): Minimum derivative above peak. Default is 4e-7.

    Returns:
        (array_like): Classification of liquid at each point: 1 = Yes,  0 = No

    """
    # search distances for potential base/top
    dheight = utils.med_diff(height)
    base_below_peak = int(np.ceil((200/dheight)))
    top_above_peak = int(np.ceil((150/dheight)))
    # init result matrices
    cloud_bit = ma.masked_all(beta.shape, dtype=int)
    # set missing values to 0
    beta_diff = np.diff(beta, axis=1).filled(fill_value=0)  # difference matrix
    beta = beta.filled(fill_value=0)
    # all peaks
    pind = scipy.signal.argrelextrema(beta, np.greater, order=4, axis=1)
    # strong peaks
    strong_peaks = np.where(beta[pind] > peak_amp)
    pind = (pind[0][strong_peaks], pind[1][strong_peaks])
    # loop over strong peaks
    for n, p in zip(*pind):
        lprof = beta[n, :]
        dprof = beta_diff[n, :]
        try:
            base = get_base_ind(dprof, p, base_below_peak, 4)
        except:
            continue
        try:
            top = get_top_ind(dprof, p, height.shape[0], top_above_peak, 4)
        except:
            continue
        tval, pval = lprof[top], lprof[p]
        # calculate peak properties
        npoints = np.count_nonzero(lprof[base:top+1])
        peak_width = height[top] - height[base]
        topder = (pval - tval) / peak_width
        if (npoints > min_points and peak_width < max_width and topder > min_top_der):
            cloud_bit[n, base:top+1] = 1
    return cloud_bit
