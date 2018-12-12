""" This module contains unit tests for utils-module. """
import sys
sys.path.append('../cloudnet')
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_almost_equal
import utils
import pytest
import atmos


def test_binning():
    """ Unit tests for units.binning_vector(). """
    arg, out = [], []
    arg.append([1, 2, 3])
    out.append([0.5, 1.5, 2.5, 3.5])
    arg.append([0.1, 0.3, 0.5])
    out.append([0.0, 0.2, 0.4, 0.6])
    arg.append([0.02, 0.04, 0.06])
    out.append([0.01, 0.03, 0.05, 0.07])
    for arg1, out1, in zip(arg, out):
        assert_array_almost_equal(utils.binning_vector(arg1), out1)


def test_bit_test():
    """ Unit tests for units.bit_test(). """
    assert utils.bit_test(0, 1) is False
    assert utils.bit_test(1, 1) is True
    assert utils.bit_test(2, 1) is False
    assert utils.bit_test(2, 2) is True

    
@pytest.mark.parametrize("n, k, res", [
    (0, 1, 1),
    (3, 1, 3),
    (4, 1, 5),
    (4, 2, 6),
])
def test_bit_set(n, k, res):
    """ Unit tests for units.bit_set(). """
    assert utils.bit_set(n, k) == res


def test_epoch():
    """ Unit tests for units.epoch2desimal_hour(). """
    n0 = 1095379200
    assert utils.epoch2desimal_hour((1970,1,1), n0) == [24]
    n1 = 12*60*60
    assert utils.epoch2desimal_hour((1970,1,1), n0 + n1) == [12]


def test_rebin():
    """ Unit tests for units.rebin_x_2d(). """
    x = np.array([1, 2, 2.99, 4, 4.99, 6, 7])
    xnew = np.array([2, 4, 6])
    data = np.array([range(1,8), range(1,8)]).T
    data_i = utils.rebin_x_2d(x, data, xnew)
    assert_array_almost_equal(data_i,np.array([[2, 4.5, 6.5],
                                               [2, 4.5, 6.5]]).T)


def test_isola():
    """ Unit tests for units.filter_isolated_pixels(). """
    x = np.array([[0,0,1,1,1],
                  [0,0,0,0,0],
                  [1,0,1,0,0],
                  [0,0,0,0,1]])                 
    x2 = np.array([[0,0,1,1,1],
                   [0,0,0,0,0],
                   [0,0,0,0,0],
                   [0,0,0,0,0]])                  
    assert_array_almost_equal(utils.filter_isolated_pixels(x), x2)


@pytest.mark.parametrize("t, res", [
    (300, 3546.1),
    (280, 995.02),
])
def test_saturation_vapor_pressure(t, res):
    """ Unit tests for atmos.saturation_vapor_pressure(). """
    cnet = atmos.saturation_vapor_pressure(t)
    assert_array_almost_equal(cnet/100, res/100, decimal=1) # 0.1hpa difference is ok


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




