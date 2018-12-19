""" This module contains general
helper functions. """

import calendar
import time
import numpy as np
import numpy.ma as ma
from scipy import stats, ndimage
from scipy.interpolate import RectBivariateSpline
# import sys


def epoch2desimal_hour(epoch, time_in):
    """Converts seconds since epoch to desimal hour of that day.

    Args:
        epoc (tuple): A 3-element tuple containing (year, month, day)
        time_in (ndarray): A 1-D array.

    Returns:
        (list): Time as desimal hour.

    Examples:
        >>> epoch2desimal_hour((1970,1,1), 1095379200) # 2004-17-10 12AM
            [24]

    """
    if not hasattr(time_in, "__iter__"):
        time_in = [time_in]
    dtime = []
    epo = calendar.timegm((*epoch, 0, 0, 0))
    for t in time_in:
        x = time.gmtime(t+epo)
        dtime.append(x.tm_hour + ((x.tm_min*60 + x.tm_sec)/3600))
    if dtime[-1] == 0:  # Last point can be 24h which would be 0 (we want 24)
        dtime[-1] = 24
    return dtime


def get_time(reso):
    """Computes fraction hour time vector 0-24 with user-given
    resolution (in seconds) where 60 is the maximum allowed value.

    Args:
        reso (int): Time resolution in seconds.

    Returns:
        (ndarray): Time vector between 0 and 24.

    Raises:
        ValueError: Bad resolution as input.

    """
    if reso < 1 or reso > 60:
        raise ValueError('Time resolution should be between 0 and 60 [s]')
    half_step = reso/7200
    return np.arange(half_step, 24-half_step, half_step*2)


def binning_vector(x):
    """Converts 1-D center points to bins with even spacing.

    Args:
        x (ndarray): A 1-D array of N real values.

    Returns:
        N + 1 edge values.

    Examples:
        >>> binning_vector([1, 2, 3])
            [0.5, 1.5, 2.5, 3.5]

    """
    edge1 = x[0] - (x[1]-x[0])/2
    edge2 = x[-1] + (x[-1]-x[-2])/2
    return np.linspace(edge1, edge2, len(x)+1)


def rebin_2d(x_in, data, x_new):
    """Rebins 2-D data in x-direction using mean. Handles masked data.

    Args:
        x_in (ndarray): A 1-D array.
        data (ndarray): 2-D input data.
        x_new (ndarray): The new x vector (center points).

    Returns:
        Rebinned (averaged) data.

    """
    edges = binning_vector(x_new)
    datai = np.zeros((len(x_new), data.shape[1]))
    data = ma.masked_invalid(data)
    for ind, values in enumerate(data.T):
        mask = ~values.mask
        if ma.any(values[mask]):
            datai[:, ind], _, _ = stats.binned_statistic(x_in[mask],
                                                         values[mask],
                                                         statistic='mean',
                                                         bins=edges)
    datai[np.isfinite(datai) == 0] = 0
    return ma.masked_equal(datai, 0)


def filter_isolated_pixels(array):
    """Returns array with completely isolated single cells removed.

    Args:
        array (ndarray): 2-D input data.

    Returns:
        Cleaned data.

    """
    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array,
                                        structure=np.ones((3, 3)))
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids+1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


def bit_test(integer, nth_bit):
    """Tests if nth bit (1,2,3..) is on for the input number.

    Args:
        integer (int): A number.
        nth_bit (int): Investigated bit.

    Returns:
        (bool): True if set, otherwise False.

    Raises:
        ValueError: negative bit as input.

    Examples:
        >>> bit_test(4, 2)
            False
        >>> bit_test(4, 3)
            True

    Notes:
        Indices start from 1 for historical reasons.

    """
    if nth_bit < 0:
        raise ValueError('Negative bit number.')
    mask = 1 << nth_bit-1
    return integer & mask > 0


def bit_set(integer, nth_bit):
    """Sets nth bit (1, 2, 3..) on input number.

    Args:
        integer: A number.
        nth_bit: Bit to be set.

    Returns:
        Integer where nth bit is set.

    Raises:
        ValueError: negative bit as input.

    Examples:
        >>> bit_set(1, 2)
            3
        >>> bit_set(0, 3)
            4

    Notes:
        Indices start from 1 for historical reasons.

    """
    if nth_bit < 0:
        raise ValueError('Negative bit number.')
    mask = 1 << nth_bit-1
    integer |= mask
    return integer


def interpolate_2d(x, y, x_new, y_new, z):
    """Linear interpolation of gridded 2d data.

    Args:
        x (ndarray): A 1-D array.
        y (ndarray): A 1-D array.
        x_new (ndarray): A 1-D array.
        y_new (ndarray): A 1-D array.
        z (ndarray): A 2-D array at points (x, y)

    Returns:
        (ndarray): Interpolated data.

    Notes:
        Does not work with nans.

    """
    fun = RectBivariateSpline(x, y, z, kx=1, ky=1)  # linear interpolation
    return fun(x_new, y_new)


def db2lin(x):
    """dB to linear conversion."""
    return 10**(x/10)


def lin2db(x):
    """Linear to dB conversion."""
    return ma.log10(x)*10


def med_diff(x):
    """Returns median difference in array."""
    return ma.median(ma.diff(x))


def bases_and_tops(y):
    """Finds islands of ones from binary array.

    From a binary vector finds all "islands" of
    ones, i.e. their starting and ending indices.

    Args:
        y (array_like): 1D array.

    Returns:
        2-element tuple containing indices of bases and tops.

    Examples:
        >>> y = [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]
        >>> bases_and_tops(y)
            ([3, 7], [4, 9])

    """
    zero = np.zeros(1)
    y2 = np.concatenate((zero, y, zero))
    y2_diff = np.diff(y2)
    bases = np.where(y2_diff == 1)[0]
    tops = np.where(y2_diff == -1)[0] - 1
    return bases, tops
