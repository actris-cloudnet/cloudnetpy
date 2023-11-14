import netCDF4
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from cloudnetpy.products import product_tools


@pytest.fixture()
def fake_categorize_file(tmpdir_factory):
    """Creates a simple categorize for testing."""
    file_name = tmpdir_factory.mktemp("data").join("categorize.nc")
    with netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC") as root_grp:
        n_points = 7
        root_grp.createDimension("time", n_points)
        var = root_grp.createVariable("time", "f8", "time")
        var[:] = np.arange(n_points)
        var = root_grp.createVariable("category_bits", "i4", "time")
        var[:] = [0, 1, 2, 4, 8, 16, 32]
        var = root_grp.createVariable("quality_bits", "i4", "time")
        var[:] = [0, 1, 2, 4, 8, 16, 32]
    return file_name


def test_category_bits(fake_categorize_file):
    obj = product_tools.CategorizeBits(fake_categorize_file)
    for key in obj.category_keys:
        assert sum(obj.category_bits[key]) == 1


def test_quality_bits(fake_categorize_file):
    obj = product_tools.CategorizeBits(fake_categorize_file)
    for key in obj.quality_keys:
        assert sum(obj.quality_bits[key]) == 1


def test_read_nc_fields(fake_categorize_file):
    assert_array_equal(
        product_tools.read_nc_fields(fake_categorize_file, "time"),
        np.arange(7),
    )
