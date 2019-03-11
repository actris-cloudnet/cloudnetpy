""" This module contains unit tests for atmos-module. """
import sys
sys.path.append('../cloudnetpy')
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_almost_equal
import pytest
from cloudnetpy import atmos


@pytest.mark.parametrize("t, res", [
    (300, 3546.1),
    (280, 995.02),
])
def test_saturation_vapor_pressure1(t, res):
    """ Unit tests for atmos.saturation_vapor_pressure(). """
    cnet = atmos.saturation_vapor_pressure(t)
    assert_array_almost_equal(cnet, res, decimal=1)


@pytest.mark.parametrize("vapor_pressure, res", [
    (500, 270.37),
    (300, 263.68),
    (200, 258.63),
    (100, 250.48),
])
def test_dew_point_temperature(vapor_pressure, res):
    """ Unit tests for atmos.dew_point(). """
    assert_array_almost_equal(atmos.dew_point_temperature(vapor_pressure), res, decimal=1)


@pytest.mark.parametrize("t, p, rh, res", [
    (280, 101330, 0.2, 273.05),
    (250, 90000, 0.01, 248.73),
])
def test_wet_bulb(t, p, rh, res):
    """ Unit tests for atmos.wet_bulb(). """
    model = {'temperature': np.array(t),
             'pressure': np.array(p),
             'rh': np.array(rh)}
    cnet = atmos.wet_bulb(model)
    assert_array_almost_equal(cnet/10, res/10, decimal=1)


