import pytest
import netCDF4
import numpy as np
import numpy.ma as ma
import numpy.testing as testing
from tests.test import initialize_test_data
import warnings

warnings.filterwarnings("ignore")

test_data_path = initialize_test_data()


@pytest.fixture
def lwc_data():
    nc_file = test_data_path + '/test_data_lwc.nc'
    data_file = netCDF4.Dataset(nc_file).variables['lwc'][:]
    return data_file[data_file == data_file.filled()] == ma.masked

    # lwc quality testing
def test_lwc_quality(lwc_data):
    # checks if array values are between wanted boundaries
    assert not lwc_data.any() >= 1e-2
    assert not lwc_data.any() <= 1e-6
