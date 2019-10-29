import numpy as np
from numpy.testing import assert_array_equal
import cloudnetpy.categorize as ds
from cloudnetpy.categorize import datasource
import pytest


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
    assert_array_equal(res, cnet)


def test_init_altitude(nc_file):
    obj = datasource.DataSource(nc_file)
    assert obj.altitude == 500


def test_getvar(nc_file, test_array):
    obj = datasource.DataSource(nc_file)
    assert_array_equal(obj.getvar('model_height'), test_array)


def test_getvar_missing(nc_file):
    obj = datasource.DataSource(nc_file)
    with pytest.raises(RuntimeError):
        obj.getvar('not_existing_variable')


def test_init_time(nc_file, test_array):
    obj = datasource.DataSource(nc_file)
    assert_array_equal(obj.time, test_array)


class FakeProfileDataSource(datasource.ProfileDataSource):
    def __init__(self, value, units):
        self.variables = {'height': Variable(value, units)}
        self.height = self._get_height()


class Variable:
    def __init__(self, value, units):
        self.value = value
        self.units = units

    def __getitem__(self, ind):
        return self.value[ind]


def test_get_height_m():
    obj = FakeProfileDataSource(np.array([1, 2, 3]), 'km')
    assert_array_equal(obj.height, [1000, 2000, 3000])


def test_get_height_km():
    obj = FakeProfileDataSource(np.array([1000, 2000, 3000]), 'm')
    assert_array_equal(obj.height, [1000, 2000, 3000])

