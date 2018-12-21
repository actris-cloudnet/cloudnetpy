""" This module contains unit tests for atmos-module. """
import sys
sys.path.append('../cloudnetpy')
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_almost_equal
import pytest
from cloudnetpy import classify


def test_get_clutter_bit():
    """ Unit tests for classify.get_clutter_bit(). """

    v = ma.array([[0.1, -0.002, 0.2, -0.3, 0.5], # (2, 5) array
                  [-0.3, 0.04, 0.4, 0.9, 1.2]])  # with two small v

    v.mask = ([[0, 0, 1, 1, 0],
               [0, 0, 0, 0, 0]])
    
    rain_bit = np.array([0, 0])

    res = np.array([[0, 1, 0, 0, 0],
                    [0, 1, 0, 0, 0]])
    
    cnet = classify.get_clutter_bit(v, rain_bit, ngates=5)
    
    assert_array_almost_equal(cnet, res)
