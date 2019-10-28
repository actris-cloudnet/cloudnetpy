import numpy as np
from numpy.testing import assert_array_equal
import cloudnetpy.categorize as ds


class FakeData(object):
    def __init__(self, arr):
        self.units = 'm'
        self.data = arr

    def __getitem__(self, item):
        return self.data


def test_km2m():
    var = FakeData(np.array([10, 20, 30, 40]))
    var.units = 'km'
    res = np.array([10000, 20000, 30000, 40000])
    cnet = ds.DataSource.km2m(var)
    assert_array_equal(res, cnet)


# Notice: assert equal even though types array items are different
def test_m2km():
    var = FakeData(np.asarray([10000, 20000, 30000, 40000], dtype=float))
    res = np.array([10, 20, 30, 40])
    cnet = ds.DataSource.m2km(var)
    print("")
    assert_array_equal(res, cnet)

