""" This module contains general helper functions. """
import logging
import os
from typing import Union, Optional, Iterator, Tuple
import uuid
import datetime
import re
import warnings
import numpy as np
import numpy.ma as ma
from scipy import stats, ndimage
from scipy.interpolate import RectBivariateSpline, griddata, RegularGridInterpolator
import pytz
import requests
from cloudnetpy.exceptions import ValidTimeStampError


SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


def seconds2hours(time_in_seconds: np.ndarray) -> np.ndarray:
    """Converts seconds since some epoch to fraction hour.

    Args:
        time_in_seconds: 1-D array of seconds since some epoch that starts on midnight.

    Returns:
        Time as fraction hour.

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
        time_in_seconds: seconds since some epoch.

    Returns:
        list: [hours, minutes, seconds] formatted as '05' etc.

    """
    seconds_since_midnight = np.mod(time_in_seconds, SECONDS_PER_DAY)
    hours = seconds_since_midnight // SECONDS_PER_HOUR
    minutes = seconds_since_midnight % SECONDS_PER_HOUR // SECONDS_PER_MINUTE
    seconds = seconds_since_midnight % SECONDS_PER_MINUTE
    time = [hours, minutes, seconds]
    return [str(t).zfill(2) for t in time]


def seconds2date(time_in_seconds: float, epoch: Optional[tuple] = (2001, 1, 1)) -> list:
    """Converts seconds since some epoch to datetime (UTC).

    Args:
        time_in_seconds: Seconds since some epoch.
        epoch: Epoch, default is (2001, 1, 1) (UTC).

    Returns:
        [year, month, day, hours, minutes, seconds] formatted as '05' etc (UTC).

    """
    epoch_in_seconds = datetime.datetime.timestamp(datetime.datetime(*epoch, tzinfo=pytz.utc))
    timestamp = time_in_seconds + epoch_in_seconds
    return datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y %m %d %H %M %S').split()


def time_grid(time_step: Optional[int] = 30) -> np.ndarray:
    """Returns decimal hour array between 0 and 24.

    Computes fraction hour time vector 0-24 with user-given
    resolution (in seconds).

    Args:
        time_step: Time resolution in seconds, greater than 1. Default is 30.

    Returns:
        Time vector between 0 and 24.

    Raises:
        ValueError: Bad resolution as input.

    """
    if time_step < 1:
        raise ValueError('Time resolution should be >= 1 seconds')
    half_step = time_step/SECONDS_PER_HOUR/2
    return np.arange(half_step, 24+half_step, half_step*2)


def binvec(x: Union[np.ndarray, list]) -> np.ndarray:
    """Converts 1-D center points to bins with even spacing.

    Args:
        x: 1-D array of N real values.

    Returns:
        ndarray: N + 1 edge values.

    Examples:
        >>> binvec([1, 2, 3])
            [0.5, 1.5, 2.5, 3.5]

    """
    edge1 = x[0] - (x[1]-x[0])/2
    edge2 = x[-1] + (x[-1]-x[-2])/2
    return np.linspace(edge1, edge2, len(x)+1)


def rebin_2d(x_in: np.ndarray,
             array: ma.MaskedArray,
             x_new: np.ndarray,
             statistic: Optional[str] = 'mean',
             n_min: Optional[int] = 1) -> Tuple[ma.MaskedArray, list]:
    """Rebins 2-D data in one dimension.

    Args:
        x_in: 1-D array with shape (n,).
        array: 2-D input data with shape (n, m).
        x_new: 1-D target vector (center points) with shape (N,).
        statistic: Statistic to be calculated. Possible statistics are 'mean', 'std'.
            Default is 'mean'.
        n_min: Minimum number of points to have good statistics in a bin. Default is 1.

    Returns:
        tuple: Rebinned data with shape (N, m) and indices of bins without enough data.

    Notes:
        0-values are masked in the returned array.

    """
    edges = binvec(x_new)
    result = np.zeros((len(x_new), array.shape[1]))
    array_screened = ma.masked_invalid(array, copy=True)  # data may contain nan-values
    for ind, values in enumerate(array_screened.T):
        mask = ~values.mask
        if ma.any(values[mask]):
            result[:, ind], _, bin_no = stats.binned_statistic(x_in[mask],
                                                               values[mask],
                                                               statistic=statistic,
                                                               bins=edges)
    result[~np.isfinite(result)] = 0
    result = ma.masked_equal(result, 0)

    # Fill bins with not enough profiles
    empty_indices = []
    for ind in range(len(edges)-1):
        is_data = np.where((x_in > edges[ind]) & (x_in <= edges[ind+1]))[0]
        if len(is_data) < n_min:
            result[ind, :] = ma.masked
            empty_indices.append(ind)
    if len(empty_indices) > 0:
        logging.info(f'No radar data in {len(empty_indices)} bins')

    return result, empty_indices


def rebin_1d(x_in: np.ndarray,
             array: ma.MaskedArray,
             x_new: np.ndarray,
             statistic: Optional[str] = 'mean') -> ma.MaskedArray:
    """Rebins 1D array.

    Args:
        x_in: 1-D array with shape (n,).
        array: 1-D input data with shape (m,).
        x_new: 1-D target vector (center points) with shape (N,).
        statistic: Statistic to be calculated. Possible statistics are 'mean', 'std'.
            Default is 'mean'.

    Returns:
        Rebinned data with shape (N,).

    """
    edges = binvec(x_new)
    result = np.zeros(len(x_new))
    array_screened = ma.masked_invalid(array, copy=True)  # data may contain nan-values
    mask = ~array_screened.mask  # pylint: disable=E1101
    if ma.any(array_screened[mask]):
        result, _, _ = stats.binned_statistic(x_in[mask],
                                              array_screened[mask],
                                              statistic=statistic,
                                              bins=edges)
    result[~np.isfinite(result)] = 0
    return ma.masked_equal(result, 0)


def filter_isolated_pixels(array: np.ndarray) -> np.ndarray:
    """From a 2D boolean array, remove completely isolated single cells.

    Args:
        array: 2-D boolean array containing isolated values.

    Returns:
        Cleaned array.

    Examples:
        >>> filter_isolated_pixels([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
            array([[0, 0, 0],
                   [0, 0, 0],
                   [0, 0, 0]])

    """
    structure = np.ones((3, 3))
    return _filter(array, structure)


def filter_x_pixels(array: np.ndarray) -> np.ndarray:
    """From a 2D boolean array, remove cells isolated in x-direction.

    Args:
        array: 2-D boolean array containing isolated pixels in x-direction.

    Returns:
        Cleaned array.

    Notes:
        Stronger cleaning than `filter_isolated_pixels()`

    Examples:
        >>> filter_x_pixels([[1, 0, 0], [0, 1, 0], [0, 1, 1]])
            array([[0, 0, 0],
                   [0, 1, 0],
                   [0, 1, 0]])

    """
    structure = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    return _filter(array, structure)


def _filter(array: np.ndarray, structure: np.ndarray) -> np.ndarray:
    filtered_array = np.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=structure)
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1))).astype(int)
    area_mask = id_sizes == 1
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


def isbit(array: np.ndarray, nth_bit: int) -> np.ndarray:
    """Tests if nth bit (0,1,2..) is set.

    Args:
        array: Integer array.
        nth_bit: Investigated bit.

    Returns:
        Boolean array denoting values where nth_bit is set.

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
        raise ValueError('Negative bit number')
    mask = 1 << nth_bit
    return array & mask > 0


def setbit(array: np.ndarray, nth_bit: int) -> int:
    """Sets nth bit (0, 1, 2..) on number.

    Args:
        array: Integer array.
        nth_bit: Bit to be set.

    Returns:
        Integer where nth bit is set.

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
        raise ValueError('Negative bit number')
    mask = 1 << nth_bit
    array |= mask
    return array


def interpolate_2d(x: np.ndarray,
                   y: np.ndarray,
                   z: np.ndarray,
                   x_new: np.ndarray,
                   y_new: np.ndarray) -> np.ndarray:
    """Linear interpolation of gridded 2d data.

    Args:
        x: 1-D array.
        y: 1-D array.
        z: 2-D array at points (x, y).
        x_new: 1-D array.
        y_new: 1-D array.

    Returns:
        Interpolated data.

    Notes:
        Does not work with nans. Ignores mask of masked data. Does not extrapolate.

    """
    fun = RectBivariateSpline(x, y, z, kx=1, ky=1)
    return fun(x_new, y_new)


def interpolate_2d_mask(x: np.ndarray,
                        y: np.ndarray,
                        z: ma.MaskedArray,
                        x_new: np.ndarray,
                        y_new: np.ndarray) -> ma.MaskedArray:
    """2D linear interpolation preserving the mask.

    Args:
        x: 1D array, x-coordinates.
        y: 1D array, y-coordinates.
        z: 2D masked array, data values.
        x_new: 1D array, new x-coordinates.
        y_new: 1D array, new y-coordinates.

    Returns:
        Interpolated 2D masked array.

    Notes:
        Points outside the original range will be nans (and masked). Uses linear interpolation.
        Input data may contain nan-values.

    """
    z = ma.masked_invalid(z, copy=True)
    # Interpolate ignoring masked values:
    valid_points = ~z.mask
    xx, yy = np.meshgrid(y, x)
    x_valid = xx[valid_points]
    y_valid = yy[valid_points]
    z_valid = z[valid_points]
    xx_new, yy_new = np.meshgrid(y_new, x_new)
    data = griddata((x_valid, y_valid), z_valid.ravel(), (xx_new, yy_new), method='linear')
    # Preserve mask:
    mask_fun = RectBivariateSpline(x, y, z.mask[:], kx=1, ky=1)
    mask = mask_fun(x_new, y_new)
    mask[mask < 0.5] = 0
    masked_array = ma.array(data, mask=mask.astype(bool))
    masked_array = ma.masked_invalid(masked_array)
    return masked_array


def interpolate_2d_nearest(x: np.ndarray,
                           y: np.ndarray,
                           z: np.ndarray,
                           x_new: np.ndarray,
                           y_new: np.ndarray) -> ma.MaskedArray:
    """2D nearest neighbor interpolation preserving mask.

    Args:
        x: 1D array, x-coordinates.
        y: 1D array, y-coordinates.
        z: 2D masked array, data values.
        x_new: 1D array, new x-coordinates.
        y_new: 1D array, new y-coordinates.

    Returns:
        Interpolated 2D masked array.

    Notes:
        Points outside the original range will be interpolated but masked.

    """
    data = ma.copy(z)
    fun = RegularGridInterpolator((x, y), data, method='nearest', bounds_error=False,
                                  fill_value=ma.masked)
    xx, yy = np.meshgrid(x_new, y_new)
    return fun((xx, yy)).T


def calc_relative_error(reference: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Calculates relative error (%)."""
    return ((array - reference) / reference) * 100


def db2lin(array: Union[float, np.ndarray], scale: Optional[int] = 10) -> np.ndarray:
    """dB to linear conversion."""
    data = array / scale
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        if ma.isMaskedArray(data):
            return ma.power(10, data)
        return np.power(10, data)


def lin2db(array: np.ndarray, scale: Optional[int] = 10) -> np.ndarray:
    """Linear to dB conversion."""
    if ma.isMaskedArray(array):
        return scale*ma.log10(array)
    return scale*np.log10(array)


def mdiff(array: np.ndarray) -> float:
    """Returns median difference of 1-D array."""
    return ma.median(ma.diff(array))


def l2norm(*args: any) -> ma.MaskedArray:
    """Returns l2 norm.

    Args:
       *args: Variable number of data (*array_like*) with the same shape.

    Returns:
        The l2 norm.

    """
    ss = 0
    for arg in args:
        if isinstance(arg, ma.MaskedArray):
            arg[~arg.mask] = arg[~arg.mask] ** 2
        else:
            arg = arg ** 2
        ss = ss + arg
    return ma.sqrt(ss)


def l2norm_weighted(values: tuple, overall_scale: float, term_weights: tuple) -> float:
    """Calculates scaled and weighted Euclidean distance.

    Calculated distance is of form: scale * sqrt((a1*a)**2 + (b1*b)**2 + ...)
    where a, b, ... are terms to be summed and a1, a2, ... are optional weights
    for the terms.

    Args:
        values: Tuple containing the values.
        overall_scale: Scale factor for the calculated Euclidean distance.
        term_weights: Weights for the terms. Must be single float or a list of numbers
            (one per term).

    Returns:
        Scaled and weighted Euclidean distance.

    TODO: Use masked arrays instead of tuples.

    """
    generic_values = ma.array(values, dtype=object)
    weighted_values = ma.multiply(generic_values, term_weights)
    return overall_scale * l2norm(*weighted_values)


def cumsumr(array: np.ndarray, axis: Optional[int] = 0) -> np.ndarray:
    """Finds cumulative sum that resets on 0.

    Args:
        array: Input array.
        axis: Axis where the sum is calculated. Default is 0.

    Returns:
        Cumulative sum, restarted at 0.

    Examples:
        >>> x = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1])
        >>> cumsumr(x)
            [0, 0, 1, 2, 0, 0, 0, 1, 2, 3]

    """
    cums = array.cumsum(axis=axis)
    return cums - np.maximum.accumulate(cums * (array == 0), axis=axis)  # pylint: disable=E1101


def ffill(array: np.ndarray, value: Optional[int] = 0) -> np.ndarray:
    """Forward fills an array.

    Args:
        array: 1-D or 2-D array.
        value: Value to be filled. Default is 0.

    Returns:
        ndarray: Forward-filled array.

    Examples:
        >>> x = np.array([0, 5, 0, 0, 2, 0])
        >>> ffill(x)
            [0, 5, 5, 5, 2, 2]

    Notes:
        Works only in axis=1 direction.

    """
    ndims = len(array.shape)
    ran = np.arange(array.shape[ndims - 1])
    idx = np.where((array != value), ran, 0)
    idx = np.maximum.accumulate(idx, axis=ndims-1)  # pylint: disable=E1101
    if ndims == 2:
        return array[np.arange(idx.shape[0])[:, None], idx]
    return array[idx]


def init(n_vars: int,
         shape: tuple,
         dtype: Optional[type] = float,
         masked: Optional[bool] = True) -> Iterator[Union[np.ndarray, ma.MaskedArray]]:
    """Initializes several numpy arrays.

    Args:
        n_vars: Number of arrays to be generated.
        shape: Shape of the arrays, e.g. (2, 3).
        dtype: The desired data-type for the arrays, e.g., int. Default is float.
        masked: If True, generated arrays are masked arrays, else ordinary numpy arrays.
            Default is True.

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
    for _ in range(n_vars):
        yield fun(shape, dtype=dtype)


def n_elements(array: np.ndarray, dist: float, var: Optional[str] = None) -> int:
    """Returns the number of elements that cover certain distance.

    Args:
        array: Input array with arbitrary units or time in fraction hour. *x* should be evenly
            spaced or at least close to.
        dist: Distance to be covered. If x is fraction time, *dist* is in minutes. Otherwise *x*
            and *dist* should have the same units.
        var: If 'time', input is fraction hour and distance in minutes, else inputs have the same
            units. Default is None (same units).

    Returns:
        Number of elements in the input array that cover *dist*.

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
    n = dist/mdiff(array)
    if var == 'time':
        n = n/60
    return int(np.round(n))


def isscalar(array: any) -> bool:
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


def get_time() -> str:
    """Returns current UTC-time."""
    return f"{datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} +00:00"


def date_range(start_date: datetime.date,
               end_date: datetime.date) -> Iterator[datetime.date]:
    """Returns range between two dates (datetimes)."""
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def get_uuid() -> str:
    """Returns unique identifier."""
    return uuid.uuid4().hex


def get_wl_band(radar_frequency: float) -> int:
    """Returns integer corresponding to radar frequency.

    Args:
        radar_frequency: Radar frequency (GHz).

    Returns:
        0 = 35GHz radar, 1 = 94Ghz radar.

    """
    return 0 if (30 < radar_frequency < 40) else 1


def get_frequency(wl_band: int) -> str:
    """Returns radar frequency string corresponding to wl band."""
    return '35.5' if wl_band == 0 else '94'


def transpose(data: np.ndarray) -> np.ndarray:
    """Transposes numpy array of (n, ) to (n, 1)."""
    if data.ndim != 1 or len(data) <= 1:
        raise ValueError('Invalid input array shape')
    return data[:, np.newaxis]


def del_dict_keys(data: dict, keys: Union[tuple, list]) -> dict:
    """Deletes multiple keys from dictionary.

    Args:
        data: A dictionary.
        keys: Keys to be deleted.

    Returns:
        Dictionary without the deleted keys.

    """
    temp_dict = data.copy()
    for key in keys:
        if key in data:
            del temp_dict[key]
    return temp_dict


def array_to_probability(array: ma.MaskedArray,
                         loc: float,
                         scale: float,
                         invert: Optional[bool] = False) -> np.ndarray:
    """Converts continuous variable into 0-1 probability.

    Args:
        array: Masked numpy array.
        loc: Center of the distribution. Values smaller than this will have small probability.
            Values greater than this will have large probability.
        scale: Width of the distribution, i.e., how fast the probability drops or increases from
            the peak.
        invert: If True, large values have small probability and vice versa. Default is False.

    Returns:
        Probability with the same shape as the input data.

    """
    arr = ma.copy(array)
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
        tilt_angle: Angle in degrees from the zenith (0 = zenith).

    Returns:
        Altitudes of the LOS points.

    Notes:
        Uses plane parallel Earth approximation.

    """
    return range_los * np.cos(np.deg2rad(tilt_angle))


def find_first_empty_line(file_name: str) -> int:
    """Finds first text file line that is empty."""
    line_number = 1
    with open(file_name) as file:
        for line in file:
            if is_empty_line(line):
                break
            line_number += 1
    return line_number


def is_empty_line(line: str) -> bool:
    """Tests if a line (of a text file) is empty."""
    if line in ('\n', '\r\n'):
        return True
    return False


def is_timestamp(timestamp: str) -> bool:
    """Tests if the input string is formatted as -yyyy-mm-dd hh:mm:ss"""
    reg_exp = re.compile(r'-\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}')
    if reg_exp.match(timestamp) is not None:
        return True
    return False


def get_sorted_filenames(file_path: str, extension: str) -> list:
    """Returns full paths of files with some extension, sorted by filename."""
    extension = extension.lower()
    all_files = os.listdir(file_path)
    files = ['/'.join((file_path, file)) for file in all_files if file.lower().endswith(extension)]
    files.sort()
    return files


def str_to_numeric(value: str) -> Union[int, float]:
    """Converts string to number (int or float)."""
    try:
        return int(value)
    except ValueError:
        return float(value)


def fetch_cloudnet_model_types() -> list:
    """Finds different model types."""
    url = f"https://cloudnet.fmi.fi/api/models"
    data = requests.get(url=url).json()
    models = [model['id'] for model in data]
    model_types = [model.split('-')[0] for model in models]
    return list(set(model_types))


def get_epoch(units: str) -> tuple:
    """Finds epoch from units string."""
    fallback = (2001, 1, 1)
    try:
        date = units.split()[2]
    except IndexError:
        return fallback
    date = date.replace(',', '')
    try:
        date_components = [int(x) for x in date.split('-')]
    except ValueError:
        try:
            date_components = [int(x) for x in date.split('.')]
        except ValueError:
            return fallback
    year, month, day = date_components
    current_year = datetime.datetime.today().year
    if (1900 < year <= current_year) and (0 < month < 13) and (0 < day < 32):
        return tuple(date_components)
    return fallback


def screen_by_time(data_in: dict, epoch: tuple, expected_date: str) -> dict:
    """"Screen data by time.

    Args:
        data_in: Dictionary containing at least 'time' key and other numpy arrays.
        epoch: Epoch of the time array, e.g., (1970, 1, 1)
        expected_date: Expected date in yyyy-mm-dd

    Returns:
        data: Screened and sorted by the time vector.

    Notes:
        - Requires 'time' key
        - Works for dimensions 1, 2, 3 (time has to be at 0-axis)
        - Does nothing for scalars

    """
    data = data_in.copy()
    valid_ind = find_valid_time_indices(data['time'], epoch, expected_date)
    n_time = len(data['time'])
    for key, array in data.items():
        if isinstance(array, list) and len(array) > 1:
            raise ValueError
        if isinstance(array, np.ndarray) and array.ndim > 0 and array.shape[0] == n_time:
            if array.ndim == 1:
                data[key] = data[key][valid_ind]
            if array.ndim == 2:
                data[key] = data[key][valid_ind, :]
            if array.ndim == 3:
                data[key] = data[key][valid_ind, :, :]
    return data


def find_valid_time_indices(time: np.ndarray, epoch: tuple, expected_date: str) -> list:
    """Finds valid time array indices for the given date.

    Args:
        time: Time in seconds from some epoch.
        epoch: Epoch of the time array, e.g., (1970, 1, 1)
        expected_date: Expected date in yyyy-mm-dd

    Returns:
        list: Valid indices for the given date in sorted order.

    Raises:
        RuntimeError: No valid timestamps.

    Examples:
        >>> time = [1, 5, 1e6, 3]
        >>> find_valid_time_indices(time, (1970, 1, 1) '1970-01-01')
            [0, 3, 2]

    """
    ind_sorted = np.argsort(time)
    ind_valid = []
    for ind in ind_sorted:
        date_str = '-'.join(seconds2date(time[ind], epoch=epoch)[:3])
        if date_str == expected_date and time[ind] not in time[ind_valid]:
            ind_valid.append(ind)
    if not ind_valid:
        raise ValidTimeStampError
    return ind_valid


def append_data(data_in: dict, key: str, array: np.ndarray) -> dict:
    """Appends data to a dictionary field (creates the field if not yet present).

    Args:
        data_in: Dictionary where data will be appended.
        key: Key of the field.
        array: Numpy array to be appended to data_in[key].

    """
    data = data_in.copy()
    if key not in data:
        data[key] = array
    else:
        data[key] = ma.concatenate((data[key], array))
    return data


def edges2mid(data: np.ndarray, reference: str) -> np.ndarray:
    """Shifts values half bin towards up or down.

    Args:
        data: 1D numpy array (e.g. range)
        reference: If 'lower', increase values by half bin. If 'upper', decrease values.

    Returns:
        Shifted values.

    """
    if reference not in ('lower', 'upper'):
        raise ValueError
    gaps = (data[1:] - data[0:-1]) / 2
    if reference == 'lower':
        gaps = np.append(gaps, gaps[-1])
        return data + gaps
    gaps = np.insert(gaps, 0, gaps[0])
    return data - gaps
