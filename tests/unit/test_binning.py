"""This module contains unit tests for utils-module."""

import numpy as np
from numpy import ma
from numpy.testing import assert_array_almost_equal, assert_array_equal

from cloudnetpy import utils


class TestRebin2D_1:
    time = np.array([1.01, 2, 2.99, 4.01, 4.99, 6.01, 6.99])
    time_new = np.array([2, 4, 6])
    data = ma.array(
        [[1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]], mask=False
    )

    def test_rebin_2d(self):
        result, empty_ind = utils.rebin_2d(self.time, self.data, self.time_new)
        expected = ma.masked_invalid([[2, 2], [4.5, 4.5], [6.5, 6.5]])
        assert_array_almost_equal(result, expected)
        assert_array_equal(result.mask, expected.mask)
        assert_array_equal(empty_ind, [])

    def test_rebin_2d_n_min(self):
        result, empty_ind = utils.rebin_2d(self.time, self.data, self.time_new, n_min=2)
        expected = ma.masked_invalid([[2, 2], [4.5, 4.5], [6.5, 6.5]])
        assert_array_almost_equal(result, expected)
        assert_array_equal(result.mask, expected.mask)
        assert_array_equal(empty_ind, [])

    def test_rebin_2d_n_min_2(self):
        result, empty_ind = utils.rebin_2d(self.time, self.data, self.time_new, n_min=3)
        expected = ma.masked_invalid([[2, 2], [np.nan, np.nan], [np.nan, np.nan]])
        assert_array_almost_equal(result, expected)
        assert_array_equal(result.mask, expected.mask)
        assert_array_equal(empty_ind, [1, 2])


class TestRebin2D_2:
    time = np.array([1, 2, 3, 4, 5])
    time_new = np.array([1.9, 3.9, 5.9])
    data = ma.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ],
        mask=False,
    )

    def test_rebin_2d_1(self):
        res, empty_bins = utils.rebin_2d(self.time, self.data, self.time_new, "max")
        expected = ma.array(
            [
                [2, 2, 2],
                [4, 4, 4],
                [5, 5, 5],
            ],
            mask=False,
        )
        assert_array_almost_equal(res, expected)
        assert_array_equal(res.mask, expected.mask)
        assert_array_equal(empty_bins, [])

    def test_rebin_2d_2(self):
        res, empty_bins = utils.rebin_2d(self.time, self.data, self.time_new, "mean")
        expected = ma.array(
            [
                [1.5, 1.5, 1.5],
                [3.5, 3.5, 3.5],
                [5.0, 5.0, 5.0],
            ],
            mask=False,
        )
        assert_array_almost_equal(res, expected)
        assert_array_equal(res.mask, expected.mask)
        assert_array_equal(empty_bins, [])


class TestRebin2DMask:
    time = np.array([1, 2, 3, 4, 5])
    time_new = np.array([1.9, 3.9, 5.9])
    data = ma.masked_invalid(
        [
            [1, 1, 1],
            [2, np.nan, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ]
    )

    def test_rebin_2d_mask_1(self):
        res, empty_bins = utils.rebin_2d(self.time, self.data, self.time_new, "max")
        expected = ma.array(
            [
                [2, 1, 2],
                [4, 4, 4],
                [5, 5, 5],
            ],
            mask=False,
        )
        assert_array_almost_equal(res, expected)
        assert_array_equal(res.mask, expected.mask)
        assert_array_equal(empty_bins, [])


def test_rebin_2d_modified_1():
    time = np.array([1, 2, 3, 4, 5])
    time_new = np.array([1.9, 3.9, 5.9])
    data = ma.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ]
    )
    res, _ = utils.rebin_2d(time, data, time_new, "max", keepdim=True)
    expected = ma.array(
        [
            [2.0, 2, 2],
            [2, 2, 2],
            [4, 4, 4],
            [4, 4, 4],
            [5, 5, 5],
        ],
        mask=False,
    )
    assert_array_almost_equal(res, expected)
    assert_array_equal(res.mask, expected.mask)


def test_rebin_2d_modified_2():
    time = np.array([1, 2, 3, 4, 5])
    time_new = np.array([2.01, 4.01, 6.01])
    data = ma.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ]
    )
    res, _ = utils.rebin_2d(time, data, time_new, "max", keepdim=True)
    expected = ma.masked_invalid(
        [
            [np.nan, np.nan, np.nan],
            [3, 3, 3],
            [3, 3, 3],
            [5, 5, 5],
            [5, 5, 5],
        ],
    )
    assert_array_almost_equal(res, expected)
    assert_array_equal(res.mask, expected.mask)


def test_rebin_2d_modified_3():
    time = np.array([1, 2, 3, 4, 5])
    time_new = np.array([10.9, 11.9, 12.9])
    data = ma.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ]
    )
    res, _ = utils.rebin_2d(time, data, time_new, "max", keepdim=True)
    expected = ma.masked_invalid(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ],
    )
    assert_array_almost_equal(res, expected)
    assert_array_equal(res.mask, expected.mask)


def test_rebin_2d_modified_4():
    time = np.array([1, 2, 3, 7, 8])
    time_new = np.array([1.9, 3.9, 5.9, 7.9, 9.9])
    data = ma.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [7, 7, 7],
            [8, 8, 8],
        ]
    )
    res, _ = utils.rebin_2d(time, data, time_new, "max", n_min=2, keepdim=True)
    expected = ma.masked_invalid(
        [
            [2, 2, 2],
            [2, 2, 2],
            [np.nan, np.nan, np.nan],
            [8, 8, 8],
            [8, 8, 8],
        ],
    )
    assert_array_almost_equal(res, expected)
    assert_array_equal(res.mask, expected.mask)


def test_rebin_2d_modified_5():
    time = np.array([1, 2, 3, 4, 5])
    time_new = np.array([1.9, 3.9, 5.9])
    data = ma.masked_invalid(
        [
            [1, np.nan, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ]
    )
    res, _ = utils.rebin_2d(time, data, time_new, "max", keepdim=True)
    expected = ma.array(
        [
            [2, 2, 2],
            [2, 2, 2],
            [4, 4, 4],
            [4, 4, 4],
            [5, 5, 5],
        ],
        mask=False,
    )
    assert_array_almost_equal(res, expected)
    assert_array_equal(res.mask, expected.mask)


def test_rebin_2d_modified_6():
    time = np.array([1, 2, 3, 4, 5])
    time_new = np.array([1.9, 3.9, 5.9])
    data = ma.masked_invalid(
        [
            [1, 1, 1],
            [2, np.nan, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ]
    )
    res, _ = utils.rebin_2d(time, data, time_new, "max", keepdim=True)
    expected = ma.array(
        [
            [2, 1, 2],
            [2, 1, 2],
            [4, 4, 4],
            [4, 4, 4],
            [5, 5, 5],
        ],
        mask=False,
    )
    assert_array_almost_equal(res, expected)
    assert_array_equal(res.mask, expected.mask)


def test_rebin_2d_modified_7():
    time = np.array([1, 2, 3, 4, 5])
    time_new = np.array([1.9, 3.9, 5.9])
    data = ma.masked_invalid(
        [
            [1, np.nan, 1],
            [2, np.nan, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ]
    )
    res, _ = utils.rebin_2d(time, data, time_new, "max", keepdim=True)
    expected = ma.masked_invalid(
        [
            [2, np.nan, 2],
            [2, np.nan, 2],
            [4, 4, 4],
            [4, 4, 4],
            [5, 5, 5],
        ],
    )
    assert_array_almost_equal(res, expected)
    assert_array_equal(res.mask, expected.mask)


def test_rebin_2d_modified_8():
    time = np.array([1, 2, 3, 4, 5])
    time_new = np.array([-12.9, -11.9, -10.9])
    data = ma.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ]
    )
    res, _ = utils.rebin_2d(time, data, time_new, "max", keepdim=True)
    expected = ma.masked_invalid(
        [
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
        ],
    )
    assert_array_almost_equal(res, expected)
    assert_array_equal(res.mask, expected.mask)


def test_rebin_2d_modified_9():
    time = np.array([3, 4, 5, 6, 7])
    time_new = np.array([1.1, 3.1, 5.1])
    data = ma.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ]
    )
    res, _ = utils.rebin_2d(time, data, time_new, "max", keepdim=True)
    expected = ma.masked_invalid(
        [
            [2, 2, 2],
            [2, 2, 2],
            [4, 4, 4],
            [4, 4, 4],
            [np.nan, np.nan, np.nan],
        ],
    )
    assert_array_almost_equal(res, expected)
    assert_array_equal(res.mask, expected.mask)


def test_rebin_2d_modified_10():
    time = np.array([3, 4, 5, 6, 7])
    time_new = np.array([4.1, 6.1, 8.1])
    data = ma.array(
        [
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3],
            [4, 4, 4],
            [5, 5, 5],
        ]
    )
    res, _ = utils.rebin_2d(time, data, time_new, "max", keepdim=True)
    expected = ma.masked_invalid(
        [
            [np.nan, np.nan, np.nan],
            [3, 3, 3],
            [3, 3, 3],
            [5, 5, 5],
            [5, 5, 5],
        ],
    )
    assert_array_almost_equal(res, expected)
    assert_array_equal(res.mask, expected.mask)
