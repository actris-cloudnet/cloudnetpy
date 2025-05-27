"""This module contains general helper functions."""

import base64
import datetime
import hashlib
import os
import re
import textwrap
import uuid
import warnings
from collections.abc import Iterator
from datetime import timezone
from typing import Any, Literal, TypeVar

import netCDF4
import numpy as np
import numpy.typing as npt
from numpy import ma
from scipy import ndimage, stats
from scipy.interpolate import RectBivariateSpline, RegularGridInterpolator, griddata

from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import SEC_IN_DAY, SEC_IN_HOUR, SEC_IN_MINUTE
from cloudnetpy.exceptions import ValidTimeStampError

Epoch = tuple[int, int, int]
Date = tuple[str, str, str]


def seconds2hours(time_in_seconds: np.ndarray) -> np.ndarray:
    """Converts seconds since some epoch to fraction hour.

    Args:
        time_in_seconds: 1-D array of seconds since some epoch that starts on midnight.

    Returns:
        Time as fraction hour.

    Notes:
        Excludes leap seconds.

    """
    seconds_since_midnight = np.mod(time_in_seconds, SEC_IN_DAY)
    fraction_hour = seconds_since_midnight / SEC_IN_HOUR
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
    seconds_since_midnight = np.mod(time_in_seconds, SEC_IN_DAY)
    hours = seconds_since_midnight // SEC_IN_HOUR
    minutes = seconds_since_midnight % SEC_IN_HOUR // SEC_IN_MINUTE
    seconds = seconds_since_midnight % SEC_IN_MINUTE
    time = [hours, minutes, seconds]
    return [str(t).zfill(2) for t in time]


def seconds2date(time_in_seconds: float, epoch: Epoch = (2001, 1, 1)) -> list:
    """Converts seconds since some epoch to datetime (UTC).

    Args:
        time_in_seconds: Seconds since some epoch.
        epoch: Epoch, default is (2001, 1, 1) (UTC).

    Returns:
        [year, month, day, hours, minutes, seconds] formatted as '05' etc (UTC).

    """
    epoch_in_seconds = datetime.datetime.timestamp(
        datetime.datetime(*epoch, tzinfo=timezone.utc),
    )
    timestamp = float(time_in_seconds) + epoch_in_seconds
    return (
        datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc)
        .strftime("%Y %m %d %H %M %S")
        .split()
    )


def datetime2decimal_hours(data: np.ndarray | list) -> np.ndarray:
    """Converts array of datetime to decimal_hours."""
    output = []
    for timestamp in data:
        t = timestamp.time()
        decimal_hours = t.hour + t.minute / SEC_IN_MINUTE + t.second / SEC_IN_HOUR
        output.append(decimal_hours)
    return np.array(output)


def time_grid(time_step: int = 30) -> np.ndarray:
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
        msg = "Time resolution should be >= 1 seconds"
        raise ValueError(msg)
    half_step = time_step / SEC_IN_HOUR / 2
    return np.arange(half_step, 24 + half_step, half_step * 2)


def binvec(x: np.ndarray | list) -> np.ndarray:
    """Converts 1-D center points to bins with even spacing.

    Args:
        x: 1-D array of N real values.

    Returns:
        ndarray: N + 1 edge values.

    Examples:
        >>> binvec([1, 2, 3])
            [0.5, 1.5, 2.5, 3.5]

    """
    edge1 = x[0] - (x[1] - x[0]) / 2
    edge2 = x[-1] + (x[-1] - x[-2]) / 2
    return np.linspace(edge1, edge2, len(x) + 1)


def rebin_2d(
    x_in: np.ndarray,
    array: np.ndarray,
    x_new: np.ndarray,
    statistic: Literal["mean", "std", "max"] = "mean",
    n_min: int = 1,
    *,
    keepdim: bool = False,
    mask_zeros: bool = False,
) -> tuple[ma.MaskedArray, np.ndarray]:
    edges = binvec(x_new)
    binn = np.digitize(x_in, edges) - 1
    n_bins = len(x_new)
    counts = np.bincount(binn[binn >= 0], minlength=n_bins)

    stat_fn: Any = {
        "mean": ma.mean,
        "std": ma.std,
        "max": ma.max,
    }[statistic]

    shape = array.shape if keepdim else (n_bins, array.shape[1])
    result: ma.MaskedArray = ma.masked_array(np.ones(shape, dtype="float32"), mask=True)

    for bin_ind in range(n_bins):
        if counts[bin_ind] < n_min:
            continue
        mask = binn == bin_ind
        block = array[mask, :]
        x_ind = mask if keepdim else bin_ind
        result[x_ind, :] = stat_fn(block, axis=0)

    empty_bins = np.where(counts < n_min)[0]

    if mask_zeros:
        result[result == 0] = ma.masked

    return result, empty_bins


def rebin_1d(
    x_in: np.ndarray,
    array: np.ndarray | ma.MaskedArray,
    x_new: np.ndarray,
    statistic: str = "mean",
) -> ma.MaskedArray:
    """Rebins 1D array.

    Args:
        x_in: 1-D array with shape (n,).
        array: 1-D input data with shape (m,).
        x_new: 1-D target vector (center points) with shape (N,).
        statistic: Statistic to be calculated. Possible statistics are 'mean', 'std'.
            Default is 'mean'.

    Returns:
        Re-binned data with shape (N,).

    """
    edges = binvec(x_new)
    result = ma.zeros(len(x_new))
    array_screened = ma.masked_invalid(array, copy=True)  # data may contain nan-values
    mask = ~array_screened.mask
    if ma.any(array_screened[mask]):
        result, _, _ = stats.binned_statistic(
            x_in[mask],
            array_screened[mask],
            statistic=statistic,
            bins=edges,
        )
    return ma.masked_invalid(result, copy=True)


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
    """Tests if nth bit (0,1,2,...) is set.

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

    See Also:
        utils.setbit()

    """
    if nth_bit < 0:
        msg = "Negative bit number"
        raise ValueError(msg)
    mask = 1 << nth_bit
    return array & mask > 0


def setbit(array: np.ndarray, nth_bit: int) -> np.ndarray:
    """Sets nth bit (0, 1, 2, ...) on number.

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

    See Also:
        utils.isbit()

    """
    if nth_bit < 0:
        msg = "Negative bit number"
        raise ValueError(msg)
    mask = 1 << nth_bit
    array |= mask
    return array


def interpolate_2d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
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


def interpolate_2d_mask(
    x: np.ndarray,
    y: np.ndarray,
    z: ma.MaskedArray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> ma.MaskedArray:
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
        Points outside the original range will be nans (and masked). Uses linear
        interpolation. Input data may contain nan-values.

    """
    z = ma.array(ma.masked_invalid(z, copy=True))
    # Interpolate ignoring masked values:
    valid_points = np.logical_not(z.mask)
    xx, yy = np.meshgrid(y, x)
    x_valid = xx[valid_points]
    y_valid = yy[valid_points]
    z_valid = z[valid_points]
    xx_new, yy_new = np.meshgrid(y_new, x_new)
    data = griddata(
        (x_valid, y_valid),
        z_valid.ravel(),
        (xx_new, yy_new),
        method="linear",
    )
    # Preserve mask:
    mask_fun = RectBivariateSpline(x, y, z.mask[:], kx=1, ky=1)
    mask = mask_fun(x_new, y_new)
    mask[mask < 0.5] = 0
    masked_array = ma.array(data, mask=mask.astype(bool))
    return ma.masked_invalid(masked_array)


def interpolate_2d_nearest(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> ma.MaskedArray:
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
    fun = RegularGridInterpolator(
        (x, y),
        data,
        method="nearest",
        bounds_error=False,
        fill_value=ma.masked,
    )
    xx, yy = np.meshgrid(x_new, y_new)
    return fun((xx, yy)).T


def calc_relative_error(reference: np.ndarray, array: np.ndarray) -> np.ndarray:
    """Calculates relative error (%)."""
    return ((array - reference) / reference) * 100


def db2lin(array: float | np.ndarray, scale: int = 10) -> np.ndarray:
    """DB to linear conversion."""
    data = array / scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if ma.isMaskedArray(data):
            return ma.power(10, data)
        return np.power(10, data)


def lin2db(array: np.ndarray, scale: int = 10) -> np.ndarray:
    """Linear to dB conversion."""
    if ma.isMaskedArray(array):
        return scale * ma.log10(array)
    return scale * np.log10(array)


def mdiff(array: np.ndarray) -> float:
    """Returns median difference of 1-D array."""
    return float(ma.median(ma.diff(array)))


def l2norm(*args) -> ma.MaskedArray:
    """Returns l2 norm.

    Args:
       *args: Variable number of data (*array_like*) with the same shape.

    Returns:
        The l2 norm.

    """
    ss = 0
    for arg in args:
        if isinstance(arg, ma.MaskedArray):
            # Raise only non-masked values, not sure if this is needed...
            arg_cpy = ma.copy(arg)
            arg_cpy[~arg.mask] = arg_cpy[~arg.mask] ** 2
        else:
            arg_cpy = arg**2
        ss = ss + arg_cpy
    return ma.sqrt(ss)


def l2norm_weighted(
    values: tuple,
    overall_scale: float,
    term_weights: tuple,
) -> ma.MaskedArray:
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


def cumsumr(array: np.ndarray, axis: int = 0) -> np.ndarray:
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
    return cums - np.maximum.accumulate(cums * (array == 0), axis=axis)


def ffill(array: np.ndarray, value: int = 0) -> np.ndarray:
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
    idx = np.maximum.accumulate(idx, axis=ndims - 1)
    if ndims == 2:
        return array[np.arange(idx.shape[0])[:, None], idx]
    return array[idx]


def init(
    n_vars: int,
    shape: tuple,
    dtype: type = float,
    *,
    masked: bool = True,
) -> Iterator[np.ndarray | ma.MaskedArray]:
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
    for _ in range(n_vars):
        if masked is True:
            yield ma.zeros(shape, dtype=dtype)
        else:
            yield np.zeros(shape, dtype=dtype)


def n_elements(array: np.ndarray, dist: float, var: str | None = None) -> int:
    """Returns the number of elements that cover certain distance.

    Args:
        array: Input array with arbitrary units or time in fraction hour. *x* should
            be evenly spaced or at least close to.
        dist: Distance to be covered. If x is fraction time, *dist* is in minutes.
            Otherwise, *x* and *dist* should have the same units.
        var: If 'time', input is fraction hour and distance in minutes, else inputs
            have the same units. Default is None (same units).

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
    n = dist / mdiff(array)
    if var == "time":
        n = n / SEC_IN_MINUTE
    return int(np.round(n))


def isscalar(array: np.ndarray | float | list | netCDF4.Variable) -> bool:
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
    return not hasattr(arr, "__len__") or arr.shape == () or len(arr) == 1


def get_time() -> str:
    """Returns current UTC-time."""
    t_zone = datetime.timezone.utc
    form = "%Y-%m-%d %H:%M:%S"
    return f"{datetime.datetime.now(tz=t_zone).strftime(form)} +00:00"


def date_range(
    start_date: datetime.date,
    end_date: datetime.date,
) -> Iterator[datetime.date]:
    """Returns range between two dates (datetimes)."""
    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def get_uuid() -> str:
    """Returns unique identifier."""
    return str(uuid.uuid4())


def get_wl_band(radar_frequency: float) -> Literal["X", "Ka", "W"]:
    """Returns IEEE radar band corresponding to radar frequency.

    Args:
        radar_frequency: Radar frequency (GHz).

    Returns:
        IEEE radar band as string.

    """
    if 8 < radar_frequency < 12:
        return "X"
    if 27 < radar_frequency < 40:
        return "Ka"
    if 75 < radar_frequency < 110:
        return "W"
    msg = f"Unknown band: {radar_frequency} GHz"
    raise ValueError(msg)


def transpose(data: np.ndarray) -> np.ndarray:
    """Transposes numpy array of (n, ) to (n, 1)."""
    if data.ndim != 1 or len(data) <= 1:
        msg = "Invalid input array shape"
        raise ValueError(msg)
    return data[:, np.newaxis]


def del_dict_keys(data: dict, keys: tuple | list) -> dict:
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


def array_to_probability(
    array: np.ndarray,
    loc: float,
    scale: float,
    *,
    invert: bool = False,
) -> np.ndarray:
    """Converts continuous variable into 0-1 probability.

    Args:
        array: Numpy array.
        loc: Center of the distribution. Values smaller than this will have small
            probability. Values greater than this will have large probability.
        scale: Width of the distribution, i.e., how fast the probability drops or
            increases from the peak.
        invert: If True, large values have small probability and vice versa.
            Default is False.

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


def is_empty_line(line: str) -> bool:
    """Tests if a line (of a text file) is empty."""
    return line in ("\n", "\r\n")


def is_timestamp(timestamp: str) -> bool:
    """Tests if the input string is formatted as -yyyy-mm-dd hh:mm:ss."""
    reg_exp = re.compile(r"-\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
    return reg_exp.match(timestamp) is not None


def get_sorted_filenames(file_path: str, extension: str) -> list:
    """Returns full paths of files with some extension, sorted by filename."""
    extension = extension.lower()
    all_files = os.listdir(file_path)
    files = [
        f"{file_path}/{file}" for file in all_files if file.lower().endswith(extension)
    ]
    files.sort()
    return files


def str_to_numeric(value: str) -> int | float:
    """Converts string to number (int or float)."""
    try:
        return int(value)
    except ValueError:
        return float(value)


def get_epoch(units: str) -> Epoch:
    """Finds epoch from units string."""
    fallback = (2001, 1, 1)
    try:
        date = units.split()[2]
    except IndexError:
        return fallback
    date = date.replace(",", "")
    if "T" in date:
        date = date[: date.index("T")]
    try:
        date_components = [int(x) for x in date.split("-")]
    except ValueError:
        try:
            date_components = [int(x) for x in date.split(".")]
        except ValueError:
            return fallback
    year, month, day = date_components
    current_year = datetime.datetime.now(tz=datetime.timezone.utc).year
    if (1900 < year <= current_year) and (0 < month < 13) and (0 < day < 32):
        return year, month, day
    return fallback


def screen_by_time(data_in: dict, epoch: Epoch, expected_date: str) -> dict:
    """Screen data by time.

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
    valid_ind = find_valid_time_indices(data["time"], epoch, expected_date)
    n_time = len(data["time"])
    for key, array in data.items():
        if isinstance(array, list) and len(array) > 1:
            raise ValueError
        if (
            isinstance(array, np.ndarray)
            and array.ndim > 0
            and array.shape[0] == n_time
        ):
            if array.ndim == 1:
                data[key] = data[key][valid_ind]
            if array.ndim == 2:
                data[key] = data[key][valid_ind, :]
            if array.ndim == 3:
                data[key] = data[key][valid_ind, :, :]
    return data


def find_valid_time_indices(time: np.ndarray, epoch: Epoch, expected_date: str) -> list:
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
    ind_valid: list[int] = []
    for ind in ind_sorted:
        date_str = "-".join(seconds2date(time[ind], epoch=epoch)[:3])
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


def edges2mid(data: np.ndarray, reference: Literal["upper", "lower"]) -> np.ndarray:
    """Shifts values half bin towards up or down.

    Args:
        data: 1D numpy array (e.g. range)
        reference: If 'lower', increase values by half bin. If 'upper', decrease values.

    Returns:
        Shifted values.

    """
    gaps = (data[1:] - data[0:-1]) / 2
    if reference == "lower":
        gaps = np.append(gaps, gaps[-1])
        return data + gaps
    gaps = np.insert(gaps, 0, gaps[0])
    return data - gaps


def get_file_type(filename: str) -> str:
    """Returns cloudnet file type from new and legacy files."""
    with netCDF4.Dataset(filename) as nc:
        if hasattr(nc, "cloudnet_file_type"):
            return nc.cloudnet_file_type
    product = filename.split("_")[-1][:-3]
    if product in ("categorize", "classification", "drizzle"):
        return product
    if product[:3] in ("lwc", "iwc"):
        return product[:3]
    msg = "Unknown file type"
    raise ValueError(msg)


def get_files_with_variables(filenames: list, variables: list[str]) -> list:
    """Returns files where all variables exist."""
    valid_files = []
    for file in filenames:
        with netCDF4.Dataset(file) as nc:
            for variable in variables:
                if variable not in nc.variables:
                    break
            else:
                valid_files.append(file)
    return valid_files


def is_all_masked(array: np.ndarray) -> bool:
    """Tests if all values are masked."""
    return ma.isMaskedArray(array) and hasattr(array, "mask") and array.mask.all()


def find_masked_profiles_indices(array: ma.MaskedArray) -> list:
    """Finds indices of masked profiles in a 2-D array."""
    non_masked_counts = np.ma.count(array, axis=1)
    masked_profiles_indices = np.where(non_masked_counts == 0)[0]
    return list(masked_profiles_indices)


T = TypeVar("T", int, str)


def _format_definition(kind: str, definitions: dict[T, str]) -> str:
    lines = [""]
    for key, value in definitions.items():
        prefix = f"{kind} {key}: "
        indent = " " * len(prefix)
        text = " ".join(value.split())
        wrapped = textwrap.wrap(prefix + text, subsequent_indent=indent)
        lines.extend(wrapped)
    return "\n".join(lines)


def status_field_definition(definitions: dict[T, str]) -> str:
    return _format_definition("Value", definitions)


def bit_field_definition(definitions: dict[T, str]) -> str:
    return _format_definition("Bit", definitions)


def path_lengths_from_ground(height: np.ndarray) -> np.ndarray:
    return np.diff(height, prepend=0)


def remove_masked_blocks(array: ma.MaskedArray, limit: int = 50) -> np.ndarray:
    """Filters out large blocks of completely masked profiles."""
    if array.ndim == 1:
        return np.array(not ma.all(array.mask))
    masked_profiles = ma.all(array.mask, axis=1)
    labeled_array, _ = ndimage.label(masked_profiles)
    mask = np.bincount(labeled_array) < limit
    mask[0] = True
    return mask[labeled_array]


def sha256sum(filename: str | os.PathLike) -> str:
    """Calculates hash of file using sha-256."""
    return _calc_hash_sum(filename, "sha256", is_base64=False)


def md5sum(filename: str | os.PathLike, *, is_base64: bool = False) -> str:
    """Calculates hash of file using md5."""
    return _calc_hash_sum(filename, "md5", is_base64=is_base64)


def _calc_hash_sum(filename, method, *, is_base64: bool) -> str:
    hash_sum = getattr(hashlib, method)()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            hash_sum.update(byte_block)
    if is_base64:
        return base64.encodebytes(hash_sum.digest()).decode("utf-8").strip()
    return hash_sum.hexdigest()


def add_site_geolocation(
    data: dict,
    *,
    gps: bool,
    site_meta: dict | None = None,
    dataset: netCDF4.Dataset | None = None,
):
    tmp_data = {}
    tmp_source = {}

    for key in ("latitude", "longitude", "altitude"):
        value = None
        source = None
        # Prefer accurate GPS coordinates. Don't trust altitude because its less
        # accurate and at least in Lindenberg BASTA there are large jumps.
        if gps and key != "altitude":
            values = None
            if isinstance(dataset, netCDF4.Dataset) and key in dataset.variables:
                values = dataset[key][:]
            elif key in data:
                values = data[key].data
            if (
                values is not None
                and not np.all(ma.getmaskarray(values))
                and np.any(values != 0)
            ):
                value = ma.masked_where(values == 0, values)
                source = "GPS"
        # User-supplied site coordinate.
        if value is None and site_meta is not None and key in site_meta:
            value = float(site_meta[key])
            source = "site coordinates"
        # From source data (CHM15k, CL61, MRR-PRO, Copernicus, Galileo...).
        # Assume value is manually set, so cannot trust it.
        if (
            value is None
            and isinstance(dataset, netCDF4.Dataset)
            and key in dataset.variables
            and not np.all(ma.getmaskarray(dataset[key][:]))
        ):
            value = dataset[key][:]
            source = "raw file"
        # From source global attributes (MIRA).
        # Seems to be manually set, so cannot trust it.
        if (
            value is None
            and isinstance(dataset, netCDF4.Dataset)
            and hasattr(dataset, key.capitalize())
        ):
            value = _parse_global_attribute_numeral(dataset, key.capitalize())
            source = "raw file"
        if value is not None:
            tmp_data[key] = value
            tmp_source[key] = source

    if "latitude" in tmp_data and "longitude" in tmp_data:
        lat = np.atleast_1d(tmp_data["latitude"])
        lon = np.atleast_1d(tmp_data["longitude"])
        lon[lon > 180] - 360
        if _are_stationary(lat, lon):
            tmp_data["latitude"] = float(ma.mean(lat))
            tmp_data["longitude"] = float(ma.mean(lon))
        else:
            tmp_data["latitude"] = lat
            tmp_data["longitude"] = lon

    if "altitude" in tmp_data:
        alt = np.atleast_1d(tmp_data["altitude"])
        if ma.max(alt) - ma.min(alt) < 100:
            tmp_data["altitude"] = float(ma.mean(alt))

    for key in ("latitude", "longitude", "altitude"):
        if key in tmp_data:
            data[key] = CloudnetArray(tmp_data[key], key, source=tmp_source[key])


def _parse_global_attribute_numeral(dataset: netCDF4.Dataset, key: str) -> float | None:
    new_str = ""
    attr = getattr(dataset, key)
    if attr == "Unknown":
        return None
    for char in attr:
        if char.isdigit() or char == ".":
            new_str += char
    return float(new_str)


def _are_stationary(latitude: npt.NDArray, longitude: npt.NDArray) -> bool:
    min_lat, max_lat = np.min(latitude), np.max(latitude)
    min_lon, max_lon = np.min(longitude), np.max(longitude)
    lat_threshold = 0.01  # deg, around 1 km
    avg_lat = (min_lat + max_lat) / 2
    lon_threshold = lat_threshold / np.cos(np.radians(avg_lat))
    lat_diff = max_lat - min_lat
    lon_diff = max_lon - min_lon
    return lat_diff <= lat_threshold and lon_diff <= lon_threshold
