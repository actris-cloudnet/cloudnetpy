""" This module contains unit tests for utils-module. """
from dataclasses import dataclass
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_almost_equal
import pytest
from cloudnetpy import utils


def test_binvec():
    """ Unit tests for units.binvec(). """
    arg, out = [], []
    arg.append([1, 2, 3])
    out.append([0.5, 1.5, 2.5, 3.5])
    arg.append([0.1, 0.3, 0.5])
    out.append([0.0, 0.2, 0.4, 0.6])
    arg.append([0.02, 0.04, 0.06])
    out.append([0.01, 0.03, 0.05, 0.07])
    for arg1, out1, in zip(arg, out):
        assert_array_almost_equal(utils.binvec(arg1), out1)


def test_isbit():
    """ Unit tests for units.isbit(). """
    assert utils.isbit(0, 0) is False
    assert utils.isbit(1, 0) is True
    assert utils.isbit(2, 0) is False
    assert utils.isbit(2, 1) is True

    
@pytest.mark.parametrize("n, k, res", [
    (0, 0, 1),
    (3, 0, 3),
    (4, 0, 5),
    (4, 1, 6),
])
def test_setbit(n, k, res):
    """ Unit tests for units.setbit(). """
    assert utils.setbit(n, k) == res


def test_seconds2hours():
    """ Unit tests for units.seconds2hour_hour(). """
    n0 = np.array([1095379200])
    assert utils.seconds2hours(n0) == [24]
    n1 = np.array([12*60*60])
    assert utils.seconds2hours(n0 + n1) == [12]


def test_rebin_2d():
    """ Unit tests for units.rebin_2d(). """
    x = np.array([1, 2, 2.99, 4, 4.99, 6, 7])
    xnew = np.array([2, 4, 6])
    data = np.array([range(1, 8), range(1, 8)]).T
    data_i = utils.rebin_2d(x, data, xnew)
    assert_array_almost_equal(data_i, np.array([[2, 4.5, 6.5],
                                                [2, 4.5, 6.5]]).T)

    data_i = utils.rebin_2d(x, data, xnew, 'std')
    arr = np.array([np.std([1, 2, 3]), np.std([4, 5]), np.std([6, 7])])
    assert_array_almost_equal(data_i, np.array([arr, arr]).T)


def test_filter_isolated_pixels():
    """ Unit tests for units.filter_isolated_pixels(). """
    x = np.array([[0, 0, 1, 1, 1],
                  [0, 0, 0, 0, 0],
                  [1, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1]])
    x2 = np.array([[0, 0, 1, 1, 1],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]])
    assert_array_almost_equal(utils.filter_isolated_pixels(x), x2)


def test_ffill():
    """Units tests for utils.ffill()."""
    x = np.array([0, 5, 0, 0, 2, 0])
    res = np.array([0, 5, 5, 5, 2, 2])
    assert_array_almost_equal(utils.ffill(x), res)

    x = np.array([[1, 0, 2, 0],
                  [0, 5, 0, 0]])
    res = np.array([[1, 1, 2, 2],
                    [0, 5, 5, 5]])
    assert_array_almost_equal(utils.ffill(x), res)

    x = np.array([[5, 1, 1, 6],
                  [3, 0, 1, 0]])
    res = np.array([[5, 5, 5, 6],
                    [3, 0, 0, 0]])
    assert_array_almost_equal(utils.ffill(x, value=1), res)


def test_cumsumr():
    """Unit tests for utils.cumsumr()."""
    x = np.array([0, 1, 2, 0, 1, 1])
    res = np.array([0, 1, 3, 0, 1, 2])
    assert_array_almost_equal(utils.cumsumr(x), res)

    x = np.array([[0, 1, 1, 0],
                  [0, 5, 0, 0]])
    res = np.array([[0, 1, 2, 0],
                    [0, 5, 0, 0]])
    assert_array_almost_equal(utils.cumsumr(x, axis=1), res)

    x = np.array([[0, 1, 1, 0],
                  [0, 5, 0, 0]])
    res = np.array([[0, 1, 1, 0],
                    [0, 6, 0, 0]])
    assert_array_almost_equal(utils.cumsumr(x, axis=0), res)

    x = np.array([[0, 1, 1, 0],
                  [0, 5, 0, 0]])
    res = np.array([[0, 1, 1, 0],
                    [0, 6, 0, 0]])
    assert_array_almost_equal(utils.cumsumr(x), res)


@pytest.mark.parametrize("input, output", [
    (np.array([1, 2, 3]), False),
    (ma.array([1, 2, 3]), False),
    (2, True),
    ((2.5,), True),
    ((2.5, 3.5), False),
    ([3], True),
    ([3, 4], False),
    (np.array(5), True),
    (ma.array(5.2), True),
    (ma.array([1, 2, 3], mask=True), False),
    (ma.array([1, 2, 3], mask=False), False),
    ([], False),
])
def test_isscalar(input, output):
    """Unit tests for utils.isscalar()."""
    assert output == utils.isscalar(input)


@dataclass
class Data:
    alt: np.array
    units: str

    def __getitem__(self, item):
        return self.alt


def test_n_elements():
    x = np.arange(1, 10)
    assert utils.n_elements(x, 5) == 5
    assert utils.n_elements(x, 5.4) == 5
    assert utils.n_elements(x, 5.5) == 6
    x = np.linspace(0, 10, 21)
    assert utils.n_elements(x, 3.5) == 7
    x = np.linspace(0, 1, 61)
    assert utils.n_elements(x, 30, 'time') == 30
    x = np.linspace(0, 6, (6*60+1)*2)
    assert utils.n_elements(x, 10, 'time') == 20


def test_l2_norm():
    """Unit tests for utils.l2_norm()"""
    x1 = np.array([2, 3])
    x2 = np.array([3, 4])
    assert_array_almost_equal(utils.l2norm(x1, x2), np.sqrt([13, 25]))
    x2m = ma.array(x2, mask=True)
    assert_array_almost_equal(utils.l2norm(x1, x2m), [2, 3])
    x2m = ma.array(x2, mask=[0, 1])
    assert_array_almost_equal(utils.l2norm(x1, x2m), [np.sqrt(13), 3])


def test_interp_2d():
    """Unit tests for utils.interp_2d()"""
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    z = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    x_new = np.array([1.5, 2.5])
    y_new = np.array([1.5, 2.5])
    result = np.array([[1.5, 1.5], [2.5, 2.5]])
    assert_array_almost_equal(utils.interpolate_2d(x, y, z, x_new, y_new),
                              result)
    x_new = np.array([1, 2])
    y_new = np.array([1, 10])
    result = np.array([[1, 1], [2, 2]])
    assert_array_almost_equal(utils.interpolate_2d(x, y, z, x_new, y_new),
                              result)
    x = ma.array([1, 2, 3])
    y = ma.array([1, 2, 3])
    z = ma.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
    z[1, 1] = ma.masked
    x_new = np.array([1.5, 2.5])
    y_new = np.array([1.5, 2.5])
    result = np.array([[1.5, 1.5], [2.5, 2.5]])
    assert_array_almost_equal(utils.interpolate_2d(x, y, z, x_new, y_new),
                              result)


def test_mdiff():
    """Unit tests for utils.mdiff()."""
    assert utils.mdiff(np.array([1, 2, 3])) == 1
    assert utils.mdiff(ma.array([1, 2, 3, 4, 5, 6], mask=[0, 1, 0, 1, 0, 0])) == 1
    assert utils.mdiff(np.array([1, 2, 10, 11, 12, 13, 14, 16])) == 1
