"""This module contains unit tests for atmos-module."""
import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_array_almost_equal, assert_array_equal

from cloudnetpy.categorize import atmos


@pytest.mark.parametrize("t, p, res", [(270, 85513, 0.001415 * 1e-3)])
def test_calc_lwc_change_rate(t, p, res):
    myres = atmos.calc_lwc_change_rate(np.array(t), np.array(p))
    assert_array_almost_equal(res, myres, decimal=4)


def test_find_cloud_bases():
    x = np.array([[0, 1, 1, 1], [0, 0, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]])
    b = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    assert_array_almost_equal(atmos.find_cloud_bases(x), b)


def test_find_cloud_tops():
    x = np.array([[0, 1, 1, 1], [0, 0, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1]])
    b = np.array([[0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 1, 0, 1]])
    assert_array_almost_equal(atmos.find_cloud_tops(x), b)


def test_find_lowest_cloud_bases():
    cloud_mask = np.array(
        [
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ],
    )
    height = np.array([1.0, 2.0, 3.0, 4.0])
    expected = ma.array([2.0, 2.0, 1.0, 1.0, 0.0, 1.0, 3.0], mask=[0, 0, 0, 0, 1, 0, 0])
    result = atmos.find_lowest_cloud_bases(cloud_mask, height)
    assert_array_almost_equal(result.data, expected.data)
    assert_array_almost_equal(result.mask, expected.mask)


def test_find_highest_cloud_tops():
    cloud_mask = np.array(
        [
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 1, 1],
        ],
    )
    height = np.array([1.0, 2.0, 3.0, 4.0])
    expected = ma.array([4.0, 3.0, 4.0, 1.0, 0.0, 4.0, 4.0], mask=[0, 0, 0, 0, 1, 0, 0])
    result = atmos.find_highest_cloud_tops(cloud_mask, height)
    assert_array_almost_equal(result.data, expected.data)
    assert_array_almost_equal(result.mask, expected.mask)


def test_distribute_lwp_to_liquid_clouds():
    lwc = np.array([[1, 1, 1], [2, 2, 2]])
    lwp = np.array([2, 8])
    result = atmos.distribute_lwp_to_liquid_clouds(lwc, lwp)
    correct = [[2 / 3, 2 / 3, 2 / 3], [16 / 6, 16 / 6, 16 / 6]]
    assert_array_equal(result, correct)
