import pytest
import numpy as np
import netCDF4
from datetime import date


DIMENSIONS = ('time', 'height', 'model_time', 'model_height')
TEST_ARRAY = np.arange(5)  # Simple array for all dimension and variables


@pytest.fixture
def test_array():
    return TEST_ARRAY


@pytest.fixture(scope='session')
def file_metadata():
    """Some example global variables to test file."""
    year, month, day = '2019', '05', '23'
    return {
        'year': year, 'month': month, 'day': day,
        'location': 'Kumpula',
        'case_date': date(int(year), int(month), int(day))
    }


@pytest.fixture(scope='session')
def nc_file(tmpdir_factory, file_metadata):
    """Creates a simple netCDF file for testing."""
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    _create_dimensions(root_grp)
    _create_dimension_variables(root_grp)
    _create_global_attributes(root_grp, file_metadata)
    _create_variable(root_grp, 'altitude', 0.5, 'km', dim=[])
    _create_variable(root_grp, 'test_array', TEST_ARRAY, 'm s-1')
    root_grp.close()
    return file_name


def _create_dimensions(root_grp):
    n_dim = len(TEST_ARRAY)
    for dim_name in DIMENSIONS:
        root_grp.createDimension(dim_name, n_dim)


def _create_dimension_variables(root_grp):
    for dim_name in DIMENSIONS:
        x = root_grp.createVariable(dim_name, 'f8', (dim_name,))
        x[:] = TEST_ARRAY


def _create_global_attributes(root_grp, meta):
    root_grp.year = meta['year']
    root_grp.month = meta['month']
    root_grp.day = meta['day']
    root_grp.location = meta['location']


def _create_variable(root_grp, name, value, units, dtype='f8', dim=('time',)):
    x = root_grp.createVariable(name, dtype, dim)
    x[:] = value
    x.units = units
