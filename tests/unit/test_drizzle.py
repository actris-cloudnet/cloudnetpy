import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from cloudnetpy.products import drizzle
import pytest


@pytest.mark.parametrize("x, result", [
    (-1000, -1),
    (-100, -0.99999),
    (-10, -0.9),
    (-1, np.exp(-1/10*np.log(10))-1),
])
def test_db2lin(x, result):
    assert_array_almost_equal(drizzle.db2lin(x), result, decimal=5)


def test_db2lin_raise():
    with pytest.raises(ValueError):
        drizzle.db2lin(150)


@pytest.mark.parametrize("x, result", [
    (1e6, 60),
    (1e5, 50),
    (1e4, 40),
])
def test_lin2db(x, result):
    assert_array_almost_equal(drizzle.lin2db(x), result, decimal=3)


def test_lin2db_raise():
    with pytest.raises(ValueError):
        drizzle.lin2db(-1)
