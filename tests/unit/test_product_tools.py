from cloudnetpy.products import product_tools
from numpy.testing import assert_array_equal


def test_read_nc_fields(nc_file, test_array):
    assert_array_equal(product_tools.read_nc_fields(nc_file, 'time'), test_array)