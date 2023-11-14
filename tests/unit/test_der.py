from collections import namedtuple

import netCDF4
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from cloudnetpy.products.der import DerSource, DropletClassification

DIMENSIONS = ("time", "height", "model_time", "model_height")
TEST_ARRAY = np.arange(3)
CategorizeBits = namedtuple("CategorizeBits", ["category_bits", "quality_bits"])
Parameters = namedtuple("Parameters", "ddBZ N dN sigma_x dsigma_x dQ")


@pytest.fixture(scope="session")
def categorize_file(tmpdir_factory, file_metadata):
    file_name = tmpdir_factory.mktemp("data").join("file.nc")

    with netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC") as nc:
        dimensions = {
            "time": 3,
            "height": 3,
            "model_time": 3,
            "model_height": 3,
        }
        for name, value in dimensions.items():
            nc.createDimension(name, value)
            var = nc.createVariable(name, "f8", name)
            var[:] = np.arange(value)
            if name == "height":
                var.units = "m"
        var = nc.createVariable("altitude", "f8")
        var[:] = 1
        var.units = "km"
        nc.createVariable("lwp", "f8", "time")[:] = [1, 1, 0.5]
        nc.createVariable("lwp_error", "f8", "time")[:] = [0.1, 0.1, 0.01]
        nc.createVariable("rainfall_rate", "i4", "time")[:] = [0, 1, 1]
        nc.createVariable("category_bits", "i4", ("time", "height"))[:] = np.array(
            [[0, 1, 0], [2, 3, 4], [4, 8, 2]],
        )
        nc.createVariable("quality_bits", "i4", ("time", "height"))[:] = np.array(
            [[0, 1, 1], [2, 3, 4], [4, 8, 1]],
        )
        nc.createVariable("temperature", "f8", ("model_time", "model_height"))[
            :
        ] = np.array([[282, 280, 278], [286, 284, 282], [284, 282, 280]])
        nc.createVariable("pressure", "f8", ("model_time", "model_height"))[
            :
        ] = np.array([[1010, 1000, 990], [1020, 1010, 1000], [1030, 1020, 1010]])
        nc.createVariable("Z", "f8", ("time", "height"))[:] = np.array(
            [[10, 20, -10], [10, 20, -10], [10, 20, -10]],
        )
    return file_name


def _create_dimensions(root_grp):
    n_dim = len(TEST_ARRAY)
    for dim_name in DIMENSIONS:
        root_grp.createDimension(dim_name, n_dim)


def _create_dimension_variables(root_grp):
    for dim_name in DIMENSIONS:
        x = root_grp.createVariable(dim_name, "f8", (dim_name,))
        x[:] = TEST_ARRAY
        if dim_name == "height":
            x.units = "m"


class TestDropletClassification:
    @pytest.fixture(autouse=True)
    def run_before_tests(self, categorize_file):
        self.obj = DropletClassification(categorize_file)

    def test_find_ice(self):
        self.obj.category_bits["falling"] = np.array([[1, 1, 1], [0, 1, 1], [0, 1, 1]])
        self.obj.category_bits["cold"] = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
        self.obj.category_bits["melting"] = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
        self.obj.category_bits["insect"] = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        expected = [[0, 0, 1], [0, 0, 0], [0, 1, 0]]
        assert_array_equal(self.obj._find_ice(), expected)

    def test_find_droplet(self):
        self.obj.category_bits["droplet"] = np.array([[1, 1, 0], [0, 1, 1], [0, 1, 1]])
        expected = [[1, 1, 0], [0, 1, 1], [0, 1, 1]]
        assert_array_equal(self.obj._find_droplet(), expected)

    def test_find_mixed(self):
        self.obj.category_bits["falling"] = np.array([[1, 1, 0], [0, 1, 1], [0, 1, 0]])
        self.obj.category_bits["droplet"] = np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
        expected = [[0, 1, 0], [0, 1, 1], [0, 1, 0]]
        assert_array_equal(self.obj._find_mixed(), expected)


class TestAppending:
    @pytest.fixture(autouse=True)
    def run_before_tests(self, categorize_file):
        self.ice_class = DropletClassification(categorize_file)
        self.der_source = DerSource(categorize_file)

    def test_append_der(self):
        self.der_source.append_der()
        expected_mask = np.array([[0, 4.19e-05, 0], [0, 4.19e-05, 0], [0, 0, 0]])
        assert_array_almost_equal(self.der_source.data["der"][:], expected_mask)
