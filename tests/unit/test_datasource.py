import numpy as np
from numpy.testing import assert_array_equal
from cloudnetpy.categorize import datasource
import pytest


def test_init_altitude(nc_file):
    obj = datasource.DataSource(nc_file)
    assert obj.altitude == 500


def test_getvar(nc_file):
    obj = datasource.DataSource(nc_file)
    assert_array_equal(obj.getvar('model_height'), np.arange(5))


def test_getvar_missing(nc_file):
    obj = datasource.DataSource(nc_file)
    with pytest.raises(RuntimeError):
        obj.getvar('not_existing_variable')


def test_init_time(nc_file):
    obj = datasource.DataSource(nc_file)
    assert_array_equal(obj.time, np.arange(5))


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
