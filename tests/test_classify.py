""" This module contains unit tests for atmos-module. """
import sys
sys.path.append('../cloudnetpy')
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_almost_equal
import pytest
from cloudnetpy import classify

def test_get_clutter_bit1():
    """ Unit tests for classify.get_clutter_bit(). """
    v = ma.array([[0.1, -0.002, 0.2, -0.3, 0.5], # (2, 5) array
                  [-0.3, 0.04, 0.4, 0.9, 1.2]])  # with two small v
    v.mask = ([[0, 0, 1, 1, 0],
               [0, 0, 0, 0, 0]])
    rain_bit = np.array([0, 0], dtype=bool)
    res = np.array([[0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0]], dtype=bool)
    cnet = classify.get_clutter_bit(v, rain_bit, ngates=5)
    assert_array_almost_equal(cnet, res)

def test_get_clutter_bit2():
    """ Unit tests for classify.get_clutter_bit(). """
    v = ma.array([[0.1, -0.002, 0.2, -0.3, 0.5], # (2, 5) array
                  [-0.3, 0.04, 0.4, 0.9, 1.2]])  # with two small v
    v.mask = ([[0, 0, 1, 1, 0],
               [0, 0, 0, 0, 0]])
    rain_bit = np.array([1, 1], dtype=bool)
    res = np.array([[0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]], dtype=bool)
    cnet = classify.get_clutter_bit(v, rain_bit, ngates=5)
    assert_array_almost_equal(cnet, res)


def test_get_falling_bit():
    """ Unit tests for classify.get_falling_bit(). """
    Z = ma.array(np.random.rand(2,5))
    Z.mask = ([[True, True, False, False, False],
               [True, False, False, False, False]])
    clutter_bit = np.zeros((2,5), dtype=bool)
    insect_bit = np.zeros((2,5), dtype=bool)
    res = np.array([[0,0,1,1,1],
                    [0,1,1,1,1]], dtype=bool)
    cnet = classify.get_falling_bit(Z, clutter_bit, insect_bit)
    assert_array_almost_equal(cnet, res)



    
