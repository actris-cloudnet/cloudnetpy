import numpy as np
import pytest
import netCDF4
from cloudnetpy.products import product_tools
from numpy.testing import assert_array_equal


@pytest.fixture
def fake_categorize_file(tmpdir_factory):
    """Creates a simple categorize for testing."""
    file_name = tmpdir_factory.mktemp("data").join("categorize.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    n_points = 7
    root_grp.createDimension('time', n_points)
    var = root_grp.createVariable('category_bits', 'i4', 'time')
    var[:] = [0, 1, 2, 4, 8, 16, 32]
    var = root_grp.createVariable('quality_bits', 'i4', 'time')
    var[:] = [0, 1, 2, 4, 8, 16, 32]
    root_grp.close()
    return file_name


def test_read_nc_fields(nc_file, test_array):
    assert_array_equal(product_tools.read_nc_fields(nc_file, 'time'), test_array)


def test_categorize_bits(fake_categorize_file):
    obj = product_tools.CategorizeBits(fake_categorize_file)
    for key in obj.category_keys:
        assert sum(obj.category_bits[key]) == 1
