""" This module contains unit tests for atmos-module. """
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from cloudnetpy.categorize import atmos


@pytest.mark.parametrize("t, p, res", [(270, 85513, 0.001415)])
def test_calc_lwc_change_rate(t, p, res):
    myres = atmos.calc_lwc_change_rate(t, p)
    assert_array_almost_equal(res, myres, decimal=4)


@pytest.mark.parametrize(
    "t, res",
    [
        (300, 3533),
        (280, 991),
    ],
)
def test_saturation_vapor_pressure(t, res):
    """Unit tests for atmos.saturation_vapor_pressure()."""
    cnet = atmos.calc_saturation_vapor_pressure(t)
    assert_array_almost_equal(cnet, res, decimal=0)


@pytest.mark.parametrize(
    "vapor_pressure, res",
    [
        (500, 270.37),
        (300, 263.68),
        (200, 258.63),
        (100, 250.48),
    ],
)
def test_dew_point_temperature(vapor_pressure, res):
    """Unit tests for atmos.dew_point()."""
    assert_array_almost_equal(atmos.calc_dew_point_temperature(vapor_pressure), res, decimal=1)


@pytest.mark.parametrize(
    "t, p, rh, res",
    [
        (280, 101330, 0.2, 273.05),
        (250, 90000, 0.01, 248.73),
    ],
)
def test_wet_bulb(t, p, rh, res):
    """Unit tests for atmos.wet_bulb()."""
    model = {"temperature": np.array(t), "pressure": np.array(p), "rh": np.array(rh)}
    cnet = atmos.calc_wet_bulb_temperature(model)
    assert_array_almost_equal(cnet / 10, res / 10, decimal=1)


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
        ]
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
        ]
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
