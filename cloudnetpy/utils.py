""" This module contains general
helper functions. """

import uuid
from datetime import datetime
import numpy as np
import numpy.ma as ma
from scipy import stats, ndimage
from scipy.interpolate import RectBivariateSpline
import requests


SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


def seconds2hours(time_in_seconds):
    """Converts seconds since some epoch to fraction hour.

    Args:
        time_in_seconds (ndarray): 1-D array of seconds since some epoch
            that starts on midnight.

    Returns:
        ndarray: Time as fraction hour.

    Notes:
        Excludes leap seconds.

    """
    seconds_since_midnight = np.mod(time_in_seconds, SECONDS_PER_DAY)
    fraction_hour = seconds_since_midnight/SECONDS_PER_HOUR
    if fraction_hour[-1] == 0:
        fraction_hour[-1] = 24
    return fraction_hour


def time_grid(time_step=30):
    """Returns decimal hour array between 0 and 24.

    Computes fraction hour time vector 0-24 with user-given
    resolution (in seconds) where 60 is the maximum allowed value.

    Args:
        time_step (int, optional): Time resolution in seconds between
            1 and 60. Default is 30.

    Returns:
        ndarray: Time vector between 0 and 24.

    Raises:
        ValueError: Bad resolution as input.

    """
    if time_step < 1 or time_step > 60:
        raise ValueError('Time resolution should be between 0 and 60 [s]')
    half_step = time_step/SECONDS_PER_HOUR/2
    return np.arange(half_step, 24+half_step, half_step*2)


def binvec(x):
    """Converts 1-D center points to bins with even spacing.

    Args:
        x (array_like): 1-D array of N real values.

    Returns:
        ndarray: N + 1 edge values.

    Examples:
        >>> binvec([1, 2, 3])
            [0.5, 1.5, 2.5, 3.5]

    """
    edge1 = x[0] - (x[1]-x[0])/2
    edge2 = x[-1] + (x[-1]-x[-2])/2
    return np.linspace(edge1, edge2, len(x)+1)


def rebin_2d(x_in, data, x_new, statistic='mean'):
    """Rebins 2-D data in one dimension using mean.

    Args:
        x_in (ndarray): 1-D array with shape (n,).
        data (MaskedArray): 2-D input data with shape (n, m).
        x_new (ndarray): 1-D target vector (center points)
            with shape (N,).
        statistic (str, optional): Statistic to be calculated. Possible
            statistics are 'mean', 'std'. Default is 'mean'.

    Returns:
        MaskedArray: Rebinned data with shape (N, m).

    Notes: 0-values are masked in the returned array.

    """
    edges = binvec(x_new)
    datai = np.zeros((len(x_new), data.shape[1]))
    data = ma.masked_invalid(data)  # data may contain nan-values
    for ind, values in enumerate(data.T):
        mask = ~values.mask
        if ma.any(values[mask]):
            datai[:, ind], _, _ = stats.binned_statistic(x_in[mask],
                                                         values[mask],
                                                         statistic=statistic,
                                                         bins=edges)
    datai[~np.isfinite(datai)] = 0
    return ma.masked_equal(datai, 0)


def rebin_1d(x_in, data, x_new, statistic='mean'):
    """Rebins 1D array."""
    edges = binvec(x_new)
    datai = np.zeros(len(x_new))
    data = ma.masked_invalid(data)  # data may contain nan-values
    mask = ~data.mask
    if ma.any(data[mask]):
        datai, _, _ = stats.binned_statistic(x_in[mask],
                                             data[mask],
                                             statistic=statistic,
                                             bins=edges)
    datai[~np.isfinite(datai)] = 0
    return ma.masked_equal(datai, 0)


def filter_isolated_pixels(array):
    """Returns array with completely isolated single cells removed.

    Args:
        array (ndarray): 2-D input data with shape.

    Returns:
        ndarray: Cleaned data.

    """
    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array,
                                        structure=np.ones((3, 3)))
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids+1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


def isbit(integer, nth_bit):
    """Tests if nth bit (0,1,2..) is on for the input number.

    Args:
        integer (int): A number.
        nth_bit (int): Investigated bit.

    Returns:
        bool: True if set, otherwise False.

    Raises:
        ValueError: negative bit as input.

    Examples:
        >>> isbit(4, 1)
            False
        >>> isbit(4, 2)
            True

    See also:
        utils.setbit()

    """
    if nth_bit < 0:
        raise ValueError('Negative bit number.')
    mask = 1 << nth_bit
    return integer & mask > 0


def setbit(integer, nth_bit):
    """Sets nth bit (0, 1, 2..) on input number.

    Args:
        integer (int): A number.
        nth_bit (int): Bit to be set.

    Returns:
        int: Integer where nth bit is set.

    Raises:
        ValueError: negative bit as input.

    Examples:
        >>> setbit(1, 1)
            3
        >>> setbit(0, 2)
            4

    See also:
        utils.isbit()

    """
    if nth_bit < 0:
        raise ValueError('Negative bit number.')
    mask = 1 << nth_bit
    integer |= mask
    return integer


def interpolate_2d(x, y, z, x_new, y_new):
    """Linear interpolation of gridded 2d data.

    Args:
        x (ndarray): 1-D array.
        y (ndarray): 1-D array.
        z (ndarray): 2-D array at points (x, y).
        x_new (ndarray): 1-D array.
        y_new (ndarray): 1-D array.

    Returns:
        ndarray: Interpolated data.

    Notes:
        Does not work with nans. Ignores mask of masked data.
        Does not extrapolate.

    """
    fun = RectBivariateSpline(x, y, z, kx=1, ky=1)
    return fun(x_new, y_new)


def interpolate_2d_masked(array, axis, axis_new):
    """Interpolates 2D array preserving the mask.

    Args:
        array (ndarray): 2D masked array.
        axis (tuple): 2-element tuple containing x and y values of the input array.
        axis_new (tuple): 2-element tuple containing new x and y values.

    Returns:
        ndarray: Interpolated 2d masked array.

    Notes:
        Uses linear interpolation.

    """
    def _mask_invalid_values(data_in):
        data_range = (np.min(array), np.max(array))
        return ma.masked_outside(data_in, *data_range)

    data_interp = interpolate_2d(*axis, array, *axis_new)
    return _mask_invalid_values(data_interp)


def calc_relative_error(reference, array):
    """Calculates relative error (%)."""
    return ((array - reference) / reference) * 100


def db2lin(x, scale=10):
    """dB to linear conversion."""
    return 10**(x/scale)


def lin2db(x, scale=10):
    """Linear to dB conversion."""
    return scale*ma.log10(x)


def mdiff(x):
    """Returns median difference of 1-D array."""
    return ma.median(ma.diff(x))


def l2norm(*args):
    """Returns l2 norm.

    Args:
       *args: Variable number of data (*array_like*)
           with the same shape.

    Returns:
        MaskedArray: The l2 norm.

    """
    ss = 0
    for arg in args:
        ss = ss + arg ** 2
    return ma.sqrt(ss)


def cumsumr(x, axis=0):
    """Finds cumulative sum that resets on 0.

    Args:
        x (ndarray): Input array.
        axis (int, optional): Axis where the sum is calculated.
            Default is 0.

    Returns:
        ndarray: Cumulative sum, restarted at 0.

    Examples:
        >>> x = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
        >>> cumsumr(x)
            [0, 0, 1, 2, 0, 0, 0, 1, 2, 3]

    """
    cums = x.cumsum(axis=axis)
    return cums - np.maximum.accumulate(cums*(x == 0), axis=axis)


def ffill(arr, value=0):
    """Forward fills a numpy array.

    Args:
        arr (ndarray): 1-D or 2-D array.
        value (int): Value to be filled. Default is 0.

    Returns:
        ndarray: Forward-filled array.

    Examples:
        >>> x = np.array([0, 5, 0, 0, 2, 0])
        >>> ffill(x)
            [0, 5, 5, 5, 2, 2]

    Notes:
        Works only in axis=1 direction.

    """
    ndims = len(arr.shape)
    ran = np.arange(arr.shape[ndims-1])
    idx = np.where((arr != value), ran, 0)
    idx = np.maximum.accumulate(idx, axis=ndims-1)
    if ndims == 2:
        return arr[np.arange(idx.shape[0])[:, None], idx]
    return arr[idx]


def init(nvars, shape, dtype=float, masked=True):
    """Initializes several numpy arrays.

    Args:
        nvars (int): Number of arrays to be generated.
        shape (tuple): Shape of the arrays, e.g. (2, 3).
        dtype (data-type, optional): The desired data-type
            for the arrays, e.g., int. Default is float.
        masked (bool): If True, generated arrays are masked
            arrays, else ordinary numpy arrays. Default is True.

    Yields:
        Iterator containing the empty arrays.

    Examples:
        >>> a, b = init(2, (2, 3))
        >>> a
            masked_array(
              data=[[0., 0., 0.],
                    [0., 0., 0.]],
              mask=False,
              fill_value=1e+20)

    """
    fun = ma.zeros if masked else np.zeros
    for _ in range(nvars):
        yield fun(shape, dtype=dtype)


def n_elements(x, dist, var=None):
    """Returns the number of elements that cover certain distance.

    Args:
        x (ndarray): Input array with arbitrary units
            or time in fraction hour. *x* should be
            evenly spaced or at least close to.
        dist (float): Distance to be covered. If x is fraction time,
            *dist* is in minutes. Otherwise *x* and *dist* should have
            the same units.
        var (str, optional): 'time' or None. If None, inputs
            have the same units. If 'time', input is fraction hour
            and distance in minutes. Default is None.

    Returns:
        int: Number of elements in the input array that cover *dist*.

    Examples:
        >>> x = np.array([2, 4, 6, 8, 10])
        >>> n_elements(x, 6)
            3

        The result is rounded to the closest integer, so:

        >>> n_elements(x, 6.9)
            3
        >>> n_elements(x, 7)
            4

        With fraction hour time vector:

        >>> x = np.linspace(0, 1, 61)
        >>> n_elements(x, 10, 'time')
            10

    """
    n = dist/mdiff(x)
    if var == 'time':
        n = n/60
    return int(np.round(n))


def isscalar(array):
    """Tests if input is scalar.

    By "scalar" we mean that array has a single value.

    Examples:
        >>> isscalar(1)
        True
        >>> isscalar([1])
        True
        >>> isscalar(np.array(1))
        True
        >>> isscalar(np.array([1]))
        True

    """
    arr = ma.array(array)
    if not hasattr(arr, '__len__') or arr.shape == () or len(arr) == 1:
        return True
    return False


def get_time():
    """Returns current UTC-time."""
    return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


def get_uuid():
    """Returns unique identifier."""
    return uuid.uuid4().hex


def get_wl_band(radar_frequency):
    """Returns integer corresponding to radar frequency.

    Args:
        radar_frequency (float): Radar frequency (GHz).

    Returns:
        int: 0=35GHz radar, 1=94Ghz radar.

    """
    return 0 if (30 < radar_frequency < 40) else 1


def transpose(x):
    """Transposes numpy array of (n, ) to (n, 1)."""
    return x[:, np.newaxis]


def del_dict_keys(dict_in, keys):
    """Removes multiple keys from dictionary."""
    for key in keys:
        if key in dict_in:
            del dict_in[key]
    return dict_in


def get_site_information(site, *args):
    """Reads site information from Cloudnet http API.

    Args:
        site (str): Site identifier, e.g. 'mace-head' or 'lindenberg'.
        args: Variable number of field names to be queried.
            Possible field names are 'latitude', 'longitude', 'altitude'
            and 'site_name'.

    Returns:
        tuple: Tuple of return values.

    Examples:
        >>> get_site_information('mace-head', 'latitude', 'longitude')

    """
    fields = ','.join(args)
    query = f"http://devcloudnet.fmi.fi/api/sites/?site={site}&fields={fields}"
    try:
        result = requests.get(query).json().values()
    except:
        result = tuple([0]*len(args))
    return result


def array_to_probability(arr_in, loc, scale, invert=False):
    """Converts continuous variable into 0-1 probability.

    Args:
        arr_in (MaskedArray): Masked numpy array.
        loc (float): Center of the distribution. Values smaller than this
            will have small probability. Values greater than this will have
            large probability.
        scale (float): Width of the distribution, i.e., how fast the probability
            drops or increases from the peak.
        invert (bool, optional): If True, large values have small
            probability and vice versa. Default is False.

    """
    arr = ma.copy(arr_in)
    prob = np.zeros(arr.shape)
    ind = ~arr.mask
    if invert:
        arr *= -1
        loc *= -1
    prob[ind] = stats.norm.cdf(arr[ind], loc=loc, scale=scale)
    return prob

