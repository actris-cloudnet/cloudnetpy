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
    """Some example global metadata to test file."""
    year, month, day = '2019', '05', '23'
    return {
        'year': year, 'month': month, 'day': day,
        'location': 'Kumpula',
        'case_date': date(int(year), int(month), int(day)),
        'altitude_km': 0.5,
    }


@pytest.fixture(scope='session')
def nc_file(tmpdir_factory, file_metadata):
    """Creates a simple netCDF file for testing."""
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    _create_dimensions(root_grp)
    _create_dimension_variables(root_grp)
    _create_global_attributes(root_grp, file_metadata)
    _create_variable(root_grp, 'altitude', file_metadata['altitude_km'], 'km', dim=[])
    _create_variable(root_grp, 'test_array', TEST_ARRAY, 'm s-1')
    _create_variable(root_grp, 'range', TEST_ARRAY, 'km')
    root_grp.close()
    return file_name


@pytest.fixture(scope='session')
def cat_file(tmpdir_factory, file_metadata):
    """Creates a simple netCDF file for testing."""
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    _create_dimensions(root_grp)
    _create_dimension_variables(root_grp)
    _create_global_attributes(root_grp, file_metadata)
    _create_variable(root_grp, 'altitude', file_metadata['altitude_km'], 'km', dim=[])
    _create_variable(root_grp, 'test_array', TEST_ARRAY, 'm s-1')
    _create_variable(root_grp, 'range', TEST_ARRAY, 'km')
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
        if dim_name == 'height':
            x.units = 'm'


def _create_global_attributes(root_grp, meta):
    for key in ('year', 'month', 'day', 'location'):
        setattr(root_grp, key, meta[key])


def _create_variable(root_grp, name, value, units, dtype='f8', dim=('time',)):
    x = root_grp.createVariable(name, dtype, dim)
    x[:] = value
    x.units = units
