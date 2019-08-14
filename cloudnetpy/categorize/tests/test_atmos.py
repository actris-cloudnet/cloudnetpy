""" This module contains unit tests for atmos-module. """
import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from cloudnetpy.categorize import atmos


@pytest.mark.parametrize("t, p, res", [
    (270, 85513, 0.001415)
])
def test_calc_lwc_change_rate(t, p, res):
    myres = atmos.calc_lwc_change_rate(t, p)
    assert_array_almost_equal(res, myres, decimal=4)


@pytest.mark.parametrize("t, res", [
    (300, 3533),
    (280, 991),
])
def test_saturation_vapor_pressure(t, res):
    """ Unit tests for atmos.saturation_vapor_pressure(). """
    cnet = atmos.calc_saturation_vapor_pressure(t)
    assert_array_almost_equal(cnet, res, decimal=0)


@pytest.mark.parametrize("vapor_pressure, res", [
    (500, 270.37),
    (300, 263.68),
    (200, 258.63),
    (100, 250.48),
])
def test_dew_point_temperature(vapor_pressure, res):
    """ Unit tests for atmos.dew_point(). """
    assert_array_almost_equal(atmos.calc_dew_point_temperature(vapor_pressure), res, decimal=1)


@pytest.mark.parametrize("t, p, rh, res", [
    (280, 101330, 0.2, 273.05),
    (250, 90000, 0.01, 248.73),
])
def test_wet_bulb(t, p, rh, res):
    """ Unit tests for atmos.wet_bulb(). """
    model = {'temperature': np.array(t),
             'pressure': np.array(p),
             'rh': np.array(rh)}
    cnet = atmos.calc_wet_bulb_temperature(model)
    assert_array_almost_equal(cnet/10, res/10, decimal=1)
