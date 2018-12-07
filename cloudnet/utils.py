""" This module contains general 
helper functions. """

import numpy as np
import numpy.ma as ma
from scipy import stats


def get_time(reso):
    """ Computes fraction hour time vector 0-24 with user-given
    resolution (in seconds) where 60 is the maximum allowed value.

    Args:
        reso (float): Time resolution in seconds.

    Returns:
        (nd.array): Time vector between 0 and 24.

    Raises:
        ValueError: Bad resolution as input.

    """
    if reso < 1 or reso > 60:
        raise ValueError('Time resolution should be between 0 and 60 [s]')
    step = reso/7200
    return np.arange(step, 24-step, step*2)


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
        x: A 1-D array of real values.
        data (nd.array): 2-D input data.
        xnew: The new x vector (center points).

    Returns:
        Rebinned field.

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
