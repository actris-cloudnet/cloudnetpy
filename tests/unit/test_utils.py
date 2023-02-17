""" This module contains unit tests for utils-module. """
import datetime
import re

import netCDF4
import numpy as np
import numpy.ma as ma
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from cloudnetpy import utils


@pytest.mark.parametrize(
    "input, output",
    [
        ([1, 2, 3], [0.5, 1.5, 2.5, 3.5]),
        ([0.1, 0.3, 0.5], [0.0, 0.2, 0.4, 0.6]),
        ([0.02, 0.04, 0.06], [0.01, 0.03, 0.05, 0.07]),
    ],
)
def test_binvec(input, output):
    assert_array_almost_equal(utils.binvec(input), output)


@pytest.mark.parametrize(
    "number, nth_bit, result",
    [
        (0, 0, False),
        (1, 0, True),
        (2, 0, False),
        (2, 1, True),
        (3, 0, True),
    ],
)
def test_isbit(number, nth_bit, result):
    assert utils.isbit(number, nth_bit) is result


@pytest.mark.parametrize(
    "n, k, res",
    [
        (0, 0, 1),
        (3, 0, 3),
        (4, 0, 5),
        (4, 1, 6),
    ],
)
def test_setbit(n, k, res):
    assert utils.setbit(n, k) == res


@pytest.mark.parametrize(
    "input, output",
    [
        ([24 * 60 * 60], 24),
        ([12 * 60 * 60], 12),
    ],
)
def test_seconds2hours(input, output):
    assert utils.seconds2hours(input) == output


@pytest.mark.parametrize(
    "input, output",
    [
        (np.array([1, 2, 3]), 1),
        (ma.array([1, 2, 3, 4, 5, 6], mask=[0, 1, 0, 1, 0, 0]), 1),
        (np.array([1, 2, 10, 11, 12, 13, 14, 16]), 1),
    ],
)
def test_mdiff(input, output):
    assert utils.mdiff(input) == output


@pytest.mark.parametrize(
    "a, b, result",
    [
        (np.array([2, 3]), np.array([3, 4]), np.sqrt([13, 25])),
        (np.array([2, 3]), ma.array([3, 4], mask=True), [2, 3]),
        (np.array([2, 3]), ma.array([3, 4], mask=[0, 1]), [np.sqrt(13), 3]),
        (
            np.array([[2, 2], [2, 2]]),
            3,
            [[np.sqrt(13), np.sqrt(13)], [np.sqrt(13), np.sqrt(13)]],
        ),
    ],
)
def test_l2_norm(a, b, result):
    assert_array_almost_equal(utils.l2norm(a, b), result)


class TestRebin2D:
    x = np.array([1.01, 2, 2.99, 4.01, 4.99, 6.01, 7])
    xnew = np.array([2, 4, 6])
    data = ma.array([range(1, 8), range(1, 8)]).T

    def test_rebin_2d(self):
        data_i, empty_ind = utils.rebin_2d(self.x, self.data, self.xnew)
        expected_data = np.array([[2, 4.5, 6.5], [2, 4.5, 6.5]]).T
        assert_array_almost_equal(data_i.data, expected_data)
        assert bool(data_i.mask) is False  # pylint: disable=E1101
        assert empty_ind == []

    def test_rebin_2d_n_min(self):
        data_i, empty_ind = utils.rebin_2d(self.x, self.data, self.xnew, n_min=2)
        expected_data = np.array([[2, 4.5, 6.5], [2, 4.5, 6.5]]).T
        assert_array_almost_equal(data_i.data, expected_data)
        assert empty_ind == []

    def test_rebin_2d_n_min_2(self):
        data_i, empty_ind = utils.rebin_2d(self.x, self.data, self.xnew, n_min=3)
        expected_data = np.array([[2, 4.5, 6.5], [2, 4.5, 6.5]]).T
        expected_mask = np.array([[0, 1, 1], [0, 1, 1]]).T
        assert_array_almost_equal(data_i.data, expected_data)
        assert_array_almost_equal(data_i.mask, expected_mask)  # pylint: disable=E1101
        assert empty_ind == [1, 2]

    def test_rebin_2d_std(self):
        data_i, empty_ind = utils.rebin_2d(self.x, self.data, self.xnew, "std")
        result = np.array([np.std([1, 2, 3]), np.std([4, 5]), np.std([6, 7])])
        result = np.array([result, result]).T
        assert_array_almost_equal(data_i, result)
        assert empty_ind == []


def test_filter_isolated_pixels():
    x = np.array([[0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [1, 0, 1, 0, 0], [0, 0, 0, 0, 1]])
    x2 = np.array([[0, 0, 1, 1, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]])
    assert_array_almost_equal(utils.filter_isolated_pixels(x), x2)


@pytest.mark.parametrize(
    "input, result",
    [
        (
            [
                [0, 1, 1, 1, 1],
                [0, 0, 0, 1, 0],
                [1, 1, 1, 0, 0],
                [0, 1, 0, 1, 1],
            ],
            [
                [0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0],
            ],
        ),
    ],
)
def test_filter_x_pixels(input, result):
    assert_array_almost_equal(utils.filter_x_pixels(input), result)


@pytest.mark.parametrize(
    "input, result",
    [
        (np.array([0, 5, 0, 0, 2, 0]), np.array([0, 5, 5, 5, 2, 2])),
        (
            np.array([[1, 0, 2, 0], [0, 5, 0, 0]]),
            np.array([[1, 1, 2, 2], [0, 5, 5, 5]]),
        ),
    ],
)
def test_ffill(input, result):
    assert_array_almost_equal(utils.ffill(input), result)


def test_ffill_2():
    x = np.array([[5, 1, 1, 6], [3, 0, 1, 0]])
    result = np.array([[5, 5, 5, 6], [3, 0, 0, 0]])
    assert_array_almost_equal(utils.ffill(x, value=1), result)


def test_cumsumr_1():
    x = np.array([0, 1, 2, 0, 1, 1])
    res = np.array([0, 1, 3, 0, 1, 2])
    assert_array_almost_equal(utils.cumsumr(x), res)


def test_cumsumr_2():
    x = np.array([[0, 1, 1, 0], [0, 5, 0, 0]])
    res = np.array([[0, 1, 2, 0], [0, 5, 0, 0]])
    assert_array_almost_equal(utils.cumsumr(x, axis=1), res)


def test_cumsumr_3():
    x = np.array([[0, 1, 1, 0], [0, 5, 0, 0]])
    res = np.array([[0, 1, 1, 0], [0, 6, 0, 0]])
    assert_array_almost_equal(utils.cumsumr(x, axis=0), res)


def test_cumsumr_4():
    x = np.array([[0, 1, 1, 0], [0, 5, 0, 0]])
    res = np.array([[0, 1, 1, 0], [0, 6, 0, 0]])
    assert_array_almost_equal(utils.cumsumr(x), res)


@pytest.mark.parametrize(
    "input, output",
    [
        (np.array([1, 2, 3]), False),
        (ma.array([1, 2, 3]), False),
        (2, True),
        ((2.5,), True),
        ((2.5, 3.5), False),
        ([3], True),
        ([3, 4], False),
        (np.array(5), True),
        (ma.array(5.2), True),
        (ma.array([1, 2, 3], mask=True), False),
        (ma.array([1, 2, 3], mask=False), False),
        ([], False),
    ],
)
def test_isscalar(input, output):
    assert output == utils.isscalar(input)


@pytest.mark.parametrize(
    "x, a, result",
    [
        (np.arange(1, 10), 5, 5),
        (np.arange(1, 10), 5.4, 5),
        (np.arange(1, 10), 5.5, 6),
        (np.arange(1, 10), 5, 5),
        (np.linspace(0, 10, 21), 3.5, 7),
    ],
)
def test_n_elements(x, a, result):
    assert utils.n_elements(x, a) == result


@pytest.mark.parametrize(
    "x, a, result",
    [
        (np.linspace(0, 1, 61), 30, 30),
        (np.linspace(0, 6, (6 * 60 + 1) * 2), 10, 20),
    ],
)
def test_n_elements_2(x, a, result):
    assert utils.n_elements(x, a, "time") == result


@pytest.mark.parametrize(
    "data, scale, weights, result",
    [
        ((2, 3), 10, (1, 2), 10 * np.sqrt([40])),
        ((np.array([1, 2]), 3), 10, (2, 3), 10 * np.sqrt([85, 97])),
    ],
)
def test_l2_norm_weighted(data, scale, weights, result):
    assert_array_almost_equal(utils.l2norm_weighted(data, scale, weights), result)


@pytest.mark.parametrize(
    "x_new, y_new, result",
    [
        (np.array([1, 2]), np.array([5, 5]), np.array([[1, 1], [1, 1]])),
        (np.array([1, 2]), np.array([5, 10]), np.array([[1, 2], [1, 2]])),
        (np.array([1.5, 2.5]), np.array([5, 10]), np.array([[1, 2], [1, 2]])),
        (
            np.array([1, 2]),
            np.array([7.5, 12.5]),
            np.array([[1.5, 2.5], [1.5, 2.5]]),
        ),
    ],
)
def test_interp_2d(x_new, y_new, result):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([5, 10, 15])
    z = np.array([5 * [1], 5 * [2], 5 * [3]]).T
    assert_array_almost_equal(utils.interpolate_2d(x, y, z, x_new, y_new), result)


@pytest.mark.parametrize(
    "x_new, y_new, expected",
    [
        (
            [1.1, 1.9],
            [10.0, 20],
            ma.array([[0.5, 1], [0.5, 1]], mask=[[0, 0], [0, 0]]),
        ),
        (
            [1.0, 2.1],
            [15.0, 25],
            ma.array([[0.75, 1.25], [np.nan, np.nan]], mask=[[0, 0], [1, 1]]),
        ),
        (
            [1.0, 10],
            [15.0, 25],
            ma.array([[0.75, 1.25], [np.nan, np.nan]], mask=[[0, 0], [1, 1]]),
        ),
        (
            [1.5, 1.9],
            [10.0, 31],
            ma.array([[0.5, np.nan], [0.5, np.nan]], mask=[[0, 1], [0, 1]]),
        ),
        (
            [1, 2],
            [9, 30],
            ma.array([[np.nan, 1.5], [np.nan, 1.5]], mask=[[1, 0], [1, 0]]),
        ),
    ],
)
def test_interpolate_2d_mask_edge(x_new, y_new, expected: ma.MaskedArray):
    x = np.array([1.0, 2, 3])
    y = np.array([10.0, 20, 30])
    z = ma.array(
        [[0.5, 1, 1.5], [0.5, 1, 1.5], [0.5, 1, 1.5]],
        mask=[[0, 0, 0], [0, 0, 0], [1, 1, 1]],
    )
    result = utils.interpolate_2d_mask(x, y, z, x_new, y_new)
    assert_array_almost_equal(result.data, expected.data)
    assert_array_almost_equal(result.mask, expected.mask)  # pylint: disable=E1101


@pytest.mark.parametrize(
    "x_new, y_new, expected",
    [
        (
            [1.1, 1.4],
            [10.0, 20],
            ma.array([[0.5, 1], [0.5, 1]], mask=[[0, 0], [0, 0]]),
        ),
        (
            [1.1, 1.6],
            [10.0, 20],
            ma.array([[0.5, 1], [0.5, 1]], mask=[[0, 0], [0, 1]]),
        ),
        (
            [1.1, 2.4],
            [9.0, 20],
            ma.array([[np.nan, 1], [np.nan, 1]], mask=[[1, 0], [1, 1]]),
        ),
        (
            [1.7, 2.3],
            [12.0, 28],
            ma.array([[0.6, 1.4], [0.6, 1.4]], mask=[[0, 0], [0, 0]]),
        ),
    ],
)
def test_interpolate_2d_mask_middle(x_new, y_new, expected):
    x = np.array([1.0, 2, 3])
    y = np.array([10.0, 20, 30])
    z = ma.array(
        [[0.5, 1, 1.5], [0.5, 1e-5, 1.5], [0.5, 1, 1.5]],
        mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    )
    result = utils.interpolate_2d_mask(x, y, z, x_new, y_new)
    assert_array_almost_equal(result.data, expected.data)
    assert_array_almost_equal(result.mask, expected.mask)  # pylint: disable=E1101


@pytest.mark.parametrize(
    "x_new, y_new, expected",
    [
        (
            np.array([1.51, 2.49]),
            np.array([10, 10]),
            ma.array([[3, 3], [3, 3]], mask=[[0, 0], [0, 0]]),
        ),
        (
            np.array([2.1, 2.9]),
            np.array([15.1, 14.9]),
            ma.array([[4, 3], [6, 5]], mask=[[0, 0], [0, 0]]),
        ),
        (
            np.array([2.9, 3.1]),
            np.array([10, 10]),
            ma.array([[5, 5], [5, 5]], mask=[[0, 0], [1, 1]]),
        ),
        (
            np.array([2.9, 2.9]),
            np.array([19, 21]),
            ma.array([[6, 6], [6, 6]], mask=[[0, 1], [0, 1]]),
        ),
    ],
)
def test_interpolate_2d_nearest(x_new, y_new, expected):
    x = np.array([1, 2, 3])
    y = np.array([10, 20])
    z = ma.array([[1, 2], [3, 4], [5, 6]], mask=False)

    result = utils.interpolate_2d_nearest(x, y, z, x_new, y_new)
    assert_array_almost_equal(expected.data, result.data)
    assert_array_equal(expected.mask, result.mask)


@pytest.mark.parametrize(
    "x_new, y_new, expected",
    [
        (
            np.array([2.4, 3.6]),
            np.array([10, 10]),
            ma.array([[3, 3], [7, 7]], mask=[[0, 0], [0, 0]]),
        ),
        (
            np.array([2.6, 3.4]),
            np.array([10, 20]),
            ma.array([[5, 6], [5, 6]], mask=[[1, 1], [1, 1]]),
        ),
    ],
)
def test_interpolate_2d_nearest_2(x_new, y_new, expected):
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([10, 20])
    z = ma.array(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
        mask=[[0, 0], [0, 0], [1, 1], [0, 0], [0, 0]],
    )

    result = utils.interpolate_2d_nearest(x, y, z, x_new, y_new)
    assert_array_almost_equal(expected.data, result.data)
    assert_array_almost_equal(expected.mask, result.mask)


class TestArrayToProbability:
    x = np.arange(11)
    loc = 5
    scale = 1
    prob = utils.array_to_probability(x, loc, scale)
    prob_inv = utils.array_to_probability(x, loc, scale, invert=True)

    def test_min(self):
        assert_array_almost_equal(self.prob[0], 0)

    def test_max(self):
        assert_array_almost_equal(self.prob[-1], 1)

    def test_min_inv(self):
        assert_array_almost_equal(self.prob_inv[-1], 0)

    def test_max_inv(self):
        assert_array_almost_equal(self.prob_inv[0], 1)


class TestDelDictKeys:
    x = {"a": 2, "b": 2, "c": 3, "d": 4}
    y = utils.del_dict_keys(x, ("a", "b"))
    assert x == {"a": 2, "b": 2, "c": 3, "d": 4}
    assert y == {"c": 3, "d": 4}


@pytest.mark.parametrize(
    "frequency, band",
    [
        (35.5, 0),
        (94, 1),
    ],
)
def test_get_wl_band(frequency, band):
    assert utils.get_wl_band(frequency) == band


@pytest.mark.parametrize(
    "reference, array, error",
    [
        (100, 110, 10),
        (1, -2, -300),
    ],
)
def test_calc_relative_error(reference, array, error):
    assert utils.calc_relative_error(reference, array) == error


def test_transpose():
    x = np.arange(10)
    x_transposed = utils.transpose(x)
    assert x.shape == (10,)
    assert x_transposed.shape == (10, 1)
    with pytest.raises(ValueError):
        utils.transpose(np.array([1]))
    with pytest.raises(ValueError):
        utils.transpose(np.array([[1, 2], [3, 5]]))


@pytest.mark.parametrize(
    "index, result",
    [
        ((0, 0), 0),
        ((4, 0), 4),
    ],
)
def test_transpose_2(index, result):
    x = np.arange(5)
    assert utils.transpose(x)[index] == result


def test_get_uuid():
    x = utils.get_uuid()
    assert isinstance(x, str)
    assert len(x) == 36


def test_get_time():
    x = utils.get_time()
    r = re.compile(".{4}-.{2}-.{2} .{2}:.{2}:.{2}")
    assert r.match(x)


@pytest.mark.parametrize(
    "los_range, tilt_angle, result",
    [
        (np.array([1, 2, 3]), 0, [1, 2, 3]),
        (np.array([1, 2, 3]), 90, [0, 0, 0]),
    ],
)
def test_range_to_height(los_range, tilt_angle, result):
    height = utils.range_to_height(los_range, tilt_angle)
    assert_array_almost_equal(height, result)


@pytest.mark.parametrize(
    "input, result",
    [
        ((1e-10,), -100),
        ((1e-10, 1), -10),
    ],
)
def test_lin2db(input, result):
    assert utils.lin2db(*input) == result


@pytest.mark.parametrize(
    "input_array, expected",
    [
        (np.array([1e-10, 1e-10]), ma.array([-100.0, -100.0], mask=[0, 0])),
        (
            ma.array([1e-10, 1e-10], mask=[0, 1]),
            ma.array([-100.0, -100.0], mask=[0, 1]),
        ),
    ],
)
def test_lin2db_arrays(input_array: np.ndarray, expected: ma.MaskedArray):
    converted = utils.lin2db(input_array)
    assert_array_equal(converted, expected)
    if ma.isMaskedArray(input_array):
        assert isinstance(converted, ma.MaskedArray)
        assert_array_equal(converted.mask, expected.mask)


@pytest.mark.parametrize(
    "data, result",
    [
        ((-100,), 1e-10),
        ((-10, 1), 1e-10),
    ],
)
def test_db2lin(data, result):
    assert utils.db2lin(*data) == result


@pytest.mark.parametrize(
    "data, expected",
    [
        (np.array([-100.0, -100.0]), ma.array([1e-10, 1e-10], mask=[0, 0])),
        (
            ma.array([-100.0, -100.0], mask=[0, 1]),
            ma.array([1e-10, 1e-10], mask=[0, 1]),
        ),
    ],
)
def test_db2lin_arrays(data: np.ndarray, expected: ma.MaskedArray):
    converted = utils.db2lin(data)
    assert_array_equal(converted, expected)
    if ma.isMaskedArray(data):
        assert isinstance(converted, ma.MaskedArray)
        assert_array_equal(converted.mask, expected.mask)  # pylint: disable=E1101


def test_time_grid():
    assert_array_equal(utils.time_grid(3600), np.linspace(0.5, 23.5, 24))


class TestRebin1D:
    x = np.array([1.01, 2, 2.99, 4.01, 4.99, 6.01, 7])
    xnew = np.array([2, 4, 6])
    data = np.arange(1, 8)

    def test_rebin_1d(self):
        data_i = utils.rebin_1d(self.x, self.data, self.xnew)
        result = np.array([2, 4.5, 6.5])
        assert_array_almost_equal(data_i, result)

    def test_rebin_1d_std(self):
        data_i = utils.rebin_1d(self.x, self.data, self.xnew, "std")
        result = np.array([np.std([1, 2, 3]), np.std([4, 5]), np.std([6, 7])])
        assert_array_almost_equal(data_i, result)


@pytest.mark.parametrize(
    "dtype",
    [
        float,
        int,
        bool,
    ],
)
def test_init(dtype):
    shape = (2, 3)
    arrays = utils.init(3, shape, dtype=dtype)
    for array in arrays:
        assert array.shape == shape
        assert array.dtype == dtype


def test_date_range():
    start_date = datetime.date(2019, 2, 27)
    end_date = datetime.date(2019, 3, 3)
    result = ["2019-02-27", "2019-02-28", "2019-03-01", "2019-03-02"]
    date_range = utils.date_range(start_date, end_date)
    for d, res in zip(date_range, result):
        assert str(d) == res


@pytest.mark.parametrize(
    "input, result",
    [
        ("A line", False),
        ("", False),
        ("\n", True),
        ("\r\n", True),
    ],
)
def test_is_empty_line(input, result):
    assert utils.is_empty_line(input) == result


@pytest.mark.parametrize(
    "input, result",
    [
        ("-2019-02-13 23:04:50", True),
        ("2019-02-13 23:04:50", False),
        ("2019-02-13", False),
    ],
)
def test_is_timestamp(input, result):
    assert utils.is_timestamp(input) == result


@pytest.mark.parametrize(
    "input, result",
    [
        (0, ["00", "00", "00"]),
        (24 * 60 * 60 - 1, ["23", "59", "59"]),
        (24 * 60 * 60 * 10, ["00", "00", "00"]),
    ],
)
def test_seconds2time(input, result):
    assert utils.seconds2time(input) == result


@pytest.mark.parametrize(
    "input, result, epoch",
    [
        (0, ["1970", "01", "01", "00", "00", "00"], (1970, 1, 1)),
        (0, ["2001", "01", "01", "00", "00", "00"], (2001, 1, 1)),
        (
            24 * 60 * 60 * 10 + 1,
            ["2001", "01", "11", "00", "00", "01"],
            (2001, 1, 1),
        ),
        (
            24 * 60 * 60 - 1,
            ["2001", "01", "01", "23", "59", "59"],
            (2001, 1, 1),
        ),
        (625107602, ["1989", "10", "23", "01", "00", "02"], (1970, 1, 1)),
        (625107602, ["1990", "10", "23", "01", "00", "02"], (1971, 1, 1)),
        (625107602, ["1995", "10", "23", "01", "00", "02"], (1976, 1, 1)),
        (625107602, ["1996", "10", "23", "01", "00", "02"], (1977, 1, 1)),
        (1545730073, ["2018", "12", "25", "09", "27", "53"], (1970, 1, 1)),
        (625107602, ["2020", "10", "23", "01", "00", "02"], (2001, 1, 1)),
        (1590278403, ["2020", "05", "24", "00", "00", "03"], (1970, 1, 1)),
    ],
)
def test_seconds2date(input, result, epoch):
    assert utils.seconds2date(input, epoch) == result
    assert result[3:] == utils.seconds2time(input)


@pytest.fixture
def example_files(tmpdir):
    file_names = ["f.LV1", "f.txt", "f.LV0", "f2.lv1", "g.LV1"]
    folder = tmpdir.mkdir("data/")
    for name in file_names:
        with open(folder.join(name), "wb") as f:
            f.write(b"abc")
    return folder


def test_get_sorted_filenames(example_files):
    dir_name = example_files.dirname + "/data"
    result = ["/".join((dir_name, x)) for x in ("f.LV1", "f2.lv1", "g.LV1")]
    assert utils.get_sorted_filenames(dir_name, ".LV1") == result


def test_fetch_cloudnet_model_types():
    model_types = utils.fetch_cloudnet_model_types()
    for model_type in ("icon", "harmonie", "ecmwf", "era5"):
        assert model_type in model_types


@pytest.mark.parametrize(
    "data, reference, expected",
    [
        (np.array([0, 1, 2, 4, 6]), "lower", np.array([0.5, 1.5, 3, 5, 7])),
        (np.array([2, 4, 8, 10]), "upper", np.array([1, 3, 6, 9])),
    ],
)
def test_edges2mid(data, reference, expected):
    assert_array_equal(utils.edges2mid(data, reference), expected)


@pytest.mark.parametrize(
    "d, key, array, expected",
    [
        ({}, "test", np.array([0.5, 1.5]), np.array([0.5, 1.5])),
        (
            {"foo": np.array([1, 2])},
            "foo",
            np.array([3, 4]),
            np.array([1, 2, 3, 4]),
        ),
        (
            {"foo": np.array([[1, 2], [1, 2], [1, 2]])},
            "foo",
            np.array([[3, 4], [3, 4]]),
            np.array([[1, 2], [1, 2], [1, 2], [3, 4], [3, 4]]),
        ),
    ],
)
def test_append_data(d, key, array, expected):
    res = utils.append_data(d, key, array)
    assert_array_equal(res[key], expected)


@pytest.mark.parametrize(
    "time, epoch, date, expected",
    [
        (
            np.array([1, 5, 1e6, 3]),
            (1970, 1, 1),
            "1970-01-01",
            np.array([0, 3, 1]),
        ),
        (
            np.array([1, 5, 2, 1e6, 3]),
            (1970, 1, 1),
            "1970-01-01",
            np.array([0, 2, 4, 1]),
        ),
        (
            np.array([1e8, 1.1, 2, 1e6, 3]),
            (1970, 1, 1),
            "1970-01-01",
            np.array([1, 2, 4]),
        ),
    ],
)
def test_find_valid_time_indices(time, epoch, date, expected):
    res = utils.find_valid_time_indices(time, epoch, date)
    assert_array_equal(res, expected)


def test_get_files_with_common_range(tmp_path):
    path1 = tmp_path / "file1.nc"
    with netCDF4.Dataset(path1, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("range", 100)
        nc.createVariable("range", dimensions="range", datatype="f8")
    path2 = tmp_path / "file2.nc"
    with netCDF4.Dataset(path2, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("range", 99)
        nc.createVariable("range", dimensions="range", datatype="f8")
    path3 = tmp_path / "file3.nc"
    with netCDF4.Dataset(path3, "w", format="NETCDF4_CLASSIC") as nc:
        nc.createDimension("range", 100)
        nc.createVariable("range", dimensions="range", datatype="f8")
    files = utils.get_files_with_common_range([path1, path2, path3])
    assert len(files) == 2
    assert path1 in files
    assert path3 in files
    assert path2 not in files
