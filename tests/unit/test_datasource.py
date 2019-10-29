import numpy as np
from numpy.testing import assert_array_equal
from cloudnetpy.categorize import datasource
import pytest


def test_init_altitude(nc_file, file_metadata):
    obj = datasource.DataSource(nc_file)
    assert obj.altitude == file_metadata['altitude_km'] * 1000


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


def test_close(nc_file):
    obj = datasource.DataSource(nc_file)
    assert obj.dataset.isopen() is True
    obj.close()
    assert obj.dataset.isopen() is False


def test_km2m(nc_file, test_array):
    obj = datasource.DataSource(nc_file)
    assert_array_equal(obj.km2m(obj.dataset.variables['range']), test_array * 1000)


def test_m2km(nc_file, test_array):
    obj = datasource.DataSource(nc_file)
    assert_array_equal(obj.m2km(obj.dataset.variables['height']), test_array / 1000)


def test_get_height_m(nc_file, test_array):
    obj = datasource.ProfileDataSource(nc_file)
    assert_array_equal(obj.height, test_array)
