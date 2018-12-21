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
    cnet = atmos.saturation_vapor_pressure(t, kind='accurate')
    assert_array_almost_equal(cnet, res, decimal=1) # 0.1hpa difference is ok


@pytest.mark.parametrize("t, res", [
    (300, 3546.1),
    (280, 995.02),
])
def test_saturation_vapor_pressure2(t, res):
    """ Unit tests for atmos.saturation_vapor_pressure(). """
    cnet = atmos.saturation_vapor_pressure(t, kind='fast')
    assert_array_almost_equal(cnet, res, decimal=1) # 0.1hpa difference is ok

    
@pytest.mark.parametrize("P_w, res", [
    (500, 270.37),
    (300, 263.68),
    (200, 258.63),
    (100, 250.48),
])
def test_dew_point(P_w, res):
    """ Unit tests for atmos.dew_point(). """
    assert_array_almost_equal(atmos.dew_point(P_w), res, decimal=1) 


@pytest.mark.parametrize("Tdry, p, rh, res", [
    (280, 101330, 0.2, 273.05),
    (250, 90000, 0.01, 248.73),
])
def test_wet_bulb(Tdry, p, rh, res):
    """ Unit tests for atmos.wet_bulb(). """
    cnet = atmos.wet_bulb(np.array(Tdry), np.array(p), np.array(rh))
    assert_array_almost_equal(cnet, res, decimal=1) 


