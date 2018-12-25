""" This module contains unit tests for atmos-module. """
import sys
sys.path.append('../cloudnetpy')
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_almost_equal
import pytest
from cloudnetpy import categorize

def test_atten_corr():
    """Unit tests for classify._correct_atten()."""
    z = ma.array([1, 2, 3, 4, 5])
    z[:3] = ma.masked
    gas_atten = np.arange(1,6)
    liq_atten = ma.array([1, 2, 3, 4, 5])
    liq_atten.mask = [0, 1, 0, 1, 0]
    
    res = ma.array([0, 0, 0, 8, 10])
    res.mask = [0, 0, 0, 1, 1]
                   
    cnet = categorize._correct_atten(z, gas_atten, liq_atten)
    assert_array_almost_equal(cnet, res)
