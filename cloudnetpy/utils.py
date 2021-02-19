""" This module contains general helper functions. """
import os
from typing import Union
import uuid
import datetime
import re
from json import JSONDecodeError
import numpy as np
import numpy.ma as ma
from scipy import stats, ndimage
from scipy.interpolate import RectBivariateSpline
import requests
import pytz


SECONDS_PER_MINUTE = 60
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


def seconds2time(time_in_seconds: float) -> list:
    """Converts seconds since some epoch to time of day.

    Args:
        time_in_seconds (float): seconds since some epoch.

    Returns:
        list: [hours, minutes, seconds] formatted as '05' etc.

    """
    seconds_since_midnight = np.mod(time_in_seconds, SECONDS_PER_DAY)
    hours = seconds_since_midnight // SECONDS_PER_HOUR
    minutes = seconds_since_midnight % SECONDS_PER_HOUR // SECONDS_PER_MINUTE
    seconds = seconds_since_midnight % SECONDS_PER_MINUTE
    time = [hours, minutes, seconds]
    return [str(t).zfill(2) for t in time]


def seconds2date(time_in_seconds: float, epoch=(2001, 1, 1)) -> list:
    """Converts seconds since some epoch to datetime (UTC).

    Args:
        time_in_seconds (float): seconds since some epoch.
        epoch (tuple, optional): Epoch, default is (2001, 1, 1) (UTC).

    Returns:
        list: [year, month, day, hours, minutes, seconds] formatted as '05' etc (UTC).

    """
    epoch_in_seconds = datetime.datetime.timestamp(datetime.datetime(*epoch, tzinfo=pytz.utc))
    timestamp = time_in_seconds + epoch_in_seconds
    return datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y %m %d %H %M %S').split()


def time_grid(time_step=30):
    """Returns decimal hour array between 0 and 24.

    Computes fraction hour time vector 0-24 with user-given
    resolution (in seconds).

    Args:
        time_step (int, optional): Time resolution in seconds, greater
            than 1. Default is 30.

    Returns:
        ndarray: Time vector between 0 and 24.

    Raises:
        ValueError: Bad resolution as input.

    """
    if time_step < 1:
        raise ValueError('Time resolution should be >= 1 seconds')
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


def rebin_2d(x_in, data, x_new, statistic='mean', n_min=1):
    """Rebins 2-D data in one dimension.

    Args:
        x_in (ndarray): 1-D array with shape (n,).
        data (MaskedArray): 2-D input data with shape (n, m).
        x_new (ndarray): 1-D target vector (center points)
            with shape (N,).
        statistic (str, optional): Statistic to be calculated. Possible
            statistics are 'mean', 'std'. Default is 'mean'.
        n_min (int): Minimum number of points to have good statistics in a bin.
            Default is 1.

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
            datai[:, ind], _, bin_no = stats.binned_statistic(x_in[mask],
                                                              values[mask],
                                                              statistic=statistic,
                                                              bins=edges)
            if n_min > 1:
                unique, counts = np.unique(bin_no, return_counts=True)
                datai[unique[counts < n_min]-1, ind] = 0

    datai[~np.isfinite(datai)] = 0
    return ma.masked_equal(datai, 0)


def rebin_1d(x_in, data, x_new, statistic='mean'):
    """Rebins 1D array.

    Args:
        x_in (ndarray): 1-D array with shape (n,).
        data (MaskedArray): 1-D input data with shape (m,).
        x_new (ndarray): 1-D target vector (center points) with shape (N,).
        statistic (str, optional): Statistic to be calculated. Possible
            statistics are 'mean', 'std'. Default is 'mean'.

    Returns:
        MaskedArray: Rebinned data with shape (N,).

    """
    edges = binvec(x_new)
    datai = np.zeros(len(x_new))
    data = ma.masked_invalid(data)  # data may contain nan-values
    mask = ~data.mask  # pylint: disable=E1101
    if ma.any(data[mask]):
        datai, _, _ = stats.binned_statistic(x_in[mask],
                                             data[mask],
                                             statistic=statistic,
                                             bins=edges)
    datai[~np.isfinite(datai)] = 0
    return ma.masked_equal(datai, 0)


def filter_isolated_pixels(array):
    """From a 2D boolean array, remove completely isolated single cells.

    Args:
        array (ndarray): 2-D boolean array (numpy array or list) containing
             isolated values.

    Returns:
        ndarray: Cleaned array.

    Examples:
        >>> filter_isolated_pixels([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])


    """
    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array,
                                        structure=np.ones((3, 3)))
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1))).astype(int)
    area_mask = id_sizes == 1
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


def filter_x_pixels(array):
    """From a 2D boolean array, remove cells isolated in x-direction.

    Args:
        array (ndarray): 2-D boolean array containing isolated pixels in x-direction.

    Returns:
        ndarray: Cleaned array.

    Examples:
        >>> filter_x_pixels([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
            array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 1, 0]])


    """
    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array,
                                        structure=np.array([[0, 1, 0],
                                                            [0, 1, 0],
                                                            [0, 1, 0]]))
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids+1))).astype(int)
    area_mask = id_sizes == 1
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


def isbit(data: np.ndarray, nth_bit: int) -> np.ndarray:
    """Tests if nth bit (0,1,2..) is on for a number.

    Args:
        data: Integer data.
        nth_bit: Investigated bit.

    Returns:
        True if set, otherwise False.

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
    return data & mask > 0


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


def interpolate_2d_masked(array, ax_values, ax_values_new):
    """Interpolates 2D array preserving the mask.

    Args:
        array (ndarray): 2D masked array.
        ax_values (tuple): 2-element tuple containing x and y values of the input array.
        ax_values_new (tuple): 2-element tuple containing new x and y values.

    Returns:
        ndarray: Interpolated 2D masked array.

    Notes:
        Uses linear interpolation.

    """
    def _mask_invalid_values(data_in):
        data_range = (np.min(array), np.max(array))
        return ma.masked_outside(data_in, *data_range)

    data_interp = interpolate_2d(*ax_values, array, *ax_values_new)
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


def l2norm_weighted(values, overall_scale, term_weights):
    """Calculates scaled and weighted Euclidean distance.

    Calculated distance is of form: scale * sqrt((a1*a)**2 + (b1*b)**2 + ...)
    where a, b, ... are terms to be summed and a1, a2, ... are optional weights
    for the terms.

    Args:
        values (tuple): Tuple containing the values.
        overall_scale (float): Scale factor for the calculated
            Euclidean distance.
        term_weights (tuple): Weights for the terms. Must be single
            float or a list of numbers (one per term).

    Returns:
        float: Scaled and weighted Euclidean distance.

    TODO: Probably better use masked arrays instead of tuples.

    """
    generic_values = ma.array(values, dtype=object)
    weighted_values = ma.multiply(generic_values, term_weights)
    return overall_scale * l2norm(*weighted_values)


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
    return cums - np.maximum.accumulate(cums*(x == 0), axis=axis)  # pylint: disable=E1101


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
    idx = np.maximum.accumulate(idx, axis=ndims-1)  # pylint: disable=E1101
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
    return datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


def date_range(start_date, end_date):
    """Return range between two dates (datetimes)."""
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


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
    """Removes multiple keys from dictionary.

    Args:
        dict_in (dict): A dictionary.
        keys (list / tuple): list of keys to be deleted.

    Returns:
        dict: Dictionary without the deleted keys.

    """
    temp_dict = dict_in.copy()
    for key in keys:
        if key in dict_in:
            del temp_dict[key]
    return temp_dict


def get_site_information(site, *args):
    """Reads site information from Cloudnet http API.

    Args:
        site (str): Site identifier, e.g. 'mace-head' or 'lindenberg'.
        args: Variable number of field names to be queried.
            Possible field names are 'latitude', 'longitude', 'altitude'
            and 'site_name'.

    Returns:
        list: List of return values as floats.

    Examples:
        >>> get_site_information('mace-head', 'latitude', 'longitude')

    """
    fields = ','.join(args)
    query = f"http://devcloudnet.fmi.fi/api/sites/?site_code={site}&fields={fields}"
    try:
        result = [*requests.get(query).json()[0]['fields'].values()]
        result = [float(x) for x in result]
        if len(result) == 1:
            result = result[0]
    except JSONDecodeError:
        result = [0]*len(args)
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

    Returns:
        ndarray: Probability.

    """
    arr = ma.copy(arr_in)
    prob = np.zeros(arr.shape)
    ind = ~arr.mask
    if invert:
        arr *= -1
        loc *= -1
    prob[ind] = stats.norm.cdf(arr[ind], loc=loc, scale=scale)
    return prob


def range_to_height(range_los: np.ndarray, tilt_angle: float) -> np.ndarray:
    """Converts distances from a tilted instrument to height above the ground.

    Args:
        range_los: Distances towards the line of sign from the instrument.
        tilt_angle: Angle in degrees from the zenith.

    Returns:
        ndarray: Altitudes of the LOS points.
    """
    return range_los * np.cos(np.deg2rad(tilt_angle))


def find_first_empty_line(file_name):
    """Finds first text file line that is empty."""
    line_number = 1
    with open(file_name) as file:
        for line in file:
            if is_empty_line(line):
                break
            line_number += 1
    return line_number


def is_empty_line(line):
    """Tests if a line (of a text file) is empty."""
    if line in ('\n', '\r\n'):
        return True
    return False


def is_timestamp(string):
    """Tests if the input string is formatted as -yyyy-mm-dd hh:mm:ss"""
    reg_exp = re.compile(r'-\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
    if reg_exp.match(string) is not None:
        return True
    return False


def get_sorted_filenames(path_to_files: str, extension: str) -> list:
    """Returns full paths of files with some extension, sorted by filename."""
    all_files = os.listdir(path_to_files)
    files = ['/'.join((path_to_files, file)) for file in all_files if file.endswith(extension)]
    files.sort()
    return files


def str_to_numeric(value: str) -> Union[int, float]:
    """Converts string to number (int or float)."""
    try:
        return int(value)
    except ValueError:
        return float(value)
