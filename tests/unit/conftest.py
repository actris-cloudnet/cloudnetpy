import pytest
import numpy as np
import netCDF4
from datetime import date

DIMENSIONS = ('time', 'height', 'model_time', 'model_height')


@pytest.fixture(scope='session')
def file_metadata():
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
    n_dim = 5
    _create_dimensions(root_grp, n_dim)
    _create_dimension_variables(root_grp, n_dim)
    _create_global_attributes(root_grp, file_metadata)
    root_grp.close()
    return file_name


def _create_dimensions(root_grp, n_dim):
    for dim_name in DIMENSIONS:
        root_grp.createDimension(dim_name, n_dim)


def _create_dimension_variables(root_grp, n_dim):
    data = np.arange(n_dim)
    for dim_name in DIMENSIONS:
        x = root_grp.createVariable(dim_name, 'f8', (dim_name,))
        x[:] = data


def _create_global_attributes(root_grp, meta):
    root_grp.year = meta['year']
    root_grp.month = meta['month']
    root_grp.day = meta['day']
    root_grp.location = meta['location']
