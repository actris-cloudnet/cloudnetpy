""" This module contains unit tests for utils-module. """
import sys
sys.path.append('../cloudnet')
from numpy.testing import assert_array_almost_equal
import utils

def test_binning():
    """ Unit test for units.binning_vector() function. """
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
    """ Unit test for units.bit_test() function. """
    assert utils.bit_test(0, 1) is False
    assert utils.bit_test(1, 1) is True
    assert utils.bit_test(2, 1) is False
    assert utils.bit_test(2, 2) is True


def test_bit_set():
    """ Unit test for units.bit_set() function. """
    assert utils.bit_set(0, 1) == 1
    assert utils.bit_set(3, 1) == 3
    assert utils.bit_set(4, 1) == 5
    assert utils.bit_set(4, 2) == 6
