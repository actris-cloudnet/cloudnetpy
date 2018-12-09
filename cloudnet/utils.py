""" This module contains general
helper functions. """

import calendar
import time
import numpy as np
import numpy.ma as ma
from scipy import stats
from scipy.interpolate import RectBivariateSpline
# import sys


def epoch2desimal_hour(epoch, time_in):
    """ Convert seconds since epoch to desimal hour of that day.

    Args:
        epoc: A 3-element tuple containing (year, month, day)
        time_in: A 1-D array.

    Returns:
        Time as desimal hour.

    """
    dtime = []
    ep = calendar.timegm((*epoch, 0, 0, 0))
    for t1 in time_in:
        x = time.gmtime(t1+ep)
        dtime.append(x.tm_hour + ((x.tm_min*60 + x.tm_sec)/3600))
    if dtime[-1] == 0:  # Last point can be 24h which would be 0 (we want 24)
        dtime[-1] = 24
    return dtime


def get_time(reso):
    """ Computes fraction hour time vector 0-24 with user-given
    resolution (in seconds) where 60 is the maximum allowed value.

    Args:
        reso: Time resolution in seconds.

    Returns:
        Time vector between 0 and 24.

    Raises:
        ValueError: Bad resolution as input.

    """
    if reso < 1 or reso > 60:
        raise ValueError('Time resolution should be between 0 and 60 [s]')
    half_step = reso/7200
    return np.arange(half_step, 24-half_step, half_step*2)


def binning_vector(x_bin):
    """ Convert 1-D (center) points to bins with even spacing.

    Args:
        x_bin: A 1-D array of N real values.

    Returns:
        N + 1 edge values.

    Examples:
        >>> binning_vector([1, 2, 3])
            [0.5, 1.5, 2.5, 3.5]

    """
    edge1 = round(x_bin[0] - (x_bin[1]-x_bin[0])/2)
    edge2 = round(x_bin[-1] + (x_bin[-1]-x_bin[-2])/2)
    return np.linspace(edge1, edge2, len(x_bin)+1)


def rebin_x_2d(x_in, data, x_new):
    """ Rebin 2D data in x-direction using mean. Handles masked data.

    Args:
        x_in: A 1-D array of real values.
        data (nd.array): 2-D input data.
        x_new: The new x vector (center points).

    Returns:
        Rebinned (averaged) data.

    """
    edges = binning_vector(x_new)
    datai = np.zeros((len(x_new), data.shape[1]))
    data = ma.masked_invalid(data)
    for ind, values in enumerate(data.T):
        mask = values.mask
        if len(values[~mask]) > 0:
            datai[:, ind], _, _ = stats.binned_statistic(x_in[~mask],
                                                         values[~mask],
                                                         statistic='mean',
                                                         bins=edges)
    datai[np.isfinite(datai) == 0] = 0
    return ma.masked_equal(datai, 0)


def filter_isolated_pixels(array):
    """ Return array with completely isolated single cells removed. """
    filtered_array = ma.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array,
                                        structure=np.ones((3, 3)))
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids+1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


def is_bit(n, k):
    """ Element-wise test of bit k (1,2,3...) on integer n. """
    mask = 1 << k-1
    return (n & mask > 0)


def set_bit(n, k):
    """ Set bit k (1,2,3..) on integer n. """
    mask = 1 << k-1
    n |= mask
    return n


def interpolate_2d(x, y, xin, yin, z):
    """ FAST interpolation of 2d data that is in grid
    Does not work with nans!
    """
    f = RectBivariateSpline(x, y, z, kx=1, ky=1)  # linear interpolation
    return f(xin, yin)
