import netCDF4
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal

from cloudnetpy.products.ier import IceClassification, IerSource

DIMENSIONS = ("time", "height", "model_time", "model_height")
TEST_ARRAY = np.arange(3)


@pytest.fixture(scope="session")
def categorize_file(tmpdir_factory, file_metadata):
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    nc = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    dimensions = {"time": 3, "height": 3, "model_time": 3, "model_height": 3}
    for name, value in dimensions.items():
        nc.createDimension(name, value)
        var = nc.createVariable(name, "f8", name)
        var[:] = np.arange(value)
        if name == "height":
            var.units = "m"
    var = nc.createVariable("altitude", "f8")
    var[:] = 1
    var.units = "km"

    nc.createVariable("radar_frequency", "f8")[:] = 35.5
    nc.createVariable("lwp", "f8", "time")[:] = [1, 1, 0.5]
    nc.createVariable("lwp_error", "f8", "time")[:] = [0.1, 0.1, 0.01]
    nc.createVariable("rain_rate", "i4", "time")[:] = [0, 1, 1]
    nc.createVariable("category_bits", "i4", ("time", "height"))[:] = np.array(
        [[0, 1, 0], [2, 3, 4], [4, 8, 2]]
    )
    nc.createVariable("quality_bits", "i4", ("time", "height"))[:] = np.array(
        [[0, 1, 1], [2, 3, 4], [4, 8, 1]]
    )
    nc.createVariable("temperature", "f8", ("model_time", "model_height"))[:] = np.array(
        [[282, 280, 278], [286, 284, 282], [284, 282, 280]]
    )
    nc.createVariable("pressure", "f8", ("model_time", "model_height"))[:] = np.array(
        [[1010, 1000, 990], [1020, 1010, 1000], [1030, 1020, 1010]]
    )
    nc.createVariable("Z", "f8", ("time", "height"))[:] = np.array(
        [[10, 20, -10], [10, 20, -10], [10, 20, -10]]
    )

    nc.close()
    return file_name


@pytest.mark.parametrize("result", ["K2liquid0", "ZT", "T", "Z", "c"])
def test_ier_coeffs(result, categorize_file):
    obj = IerSource(categorize_file, "ier")
    assert result in obj.coefficients._fields
    assert obj.coefficients == (0.878, -0.000205, -0.0015, 0.0016, -1.52)


def test_ier_temperature(categorize_file):
    obj = IerSource(categorize_file, "ier")
    expected = [[8.85, 6.85, 4.85], [12.85, 10.85, 8.85], [10.85, 8.85, 6.85]]
    assert_array_almost_equal(obj.temperature, expected)


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


class TestIceClassification:
    @pytest.fixture(autouse=True)
    def run_before_tests(self, categorize_file):
        self.obj = IceClassification(categorize_file)

    def test_find_ice(self):
        self.obj.category_bits["falling"] = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1])
        self.obj.category_bits["cold"] = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
        self.obj.category_bits["melting"] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.obj.category_bits["insect"] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
        expected = [0, 1, 1, 0, 1, 0, 0, 1, 0]
        assert_array_almost_equal(self.obj._find_ice(), expected)

    def test_would_be_ice(self):
        self.obj.category_bits["falling"] = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1])
        self.obj.category_bits["cold"] = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
        self.obj.category_bits["melting"] = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1])
        self.obj.category_bits["insect"] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])
        expected = [0, 1, 0, 1, 1, 0, 0, 1, 1]
        assert_array_almost_equal(self.obj._find_would_be_ice(), expected)

    def test_find_corrected_ice(self):
        self.obj.is_ice = np.array([1, 1, 1, 1, 0, 0])
        self.obj.quality_bits["attenuated"] = np.array([1, 1, 1, 0, 1, 0])
        self.obj.quality_bits["corrected"] = np.array([1, 0, 0, 1, 1, 0])
        expected = [1, 0, 0, 0, 0, 0]
        assert_array_almost_equal(self.obj._find_corrected_ice(), expected)

    def test_find_uncorrected_ice(self):
        self.obj.is_ice = np.array([1, 1, 1, 1, 0, 0])
        self.obj.quality_bits["attenuated"] = np.array([1, 1, 0, 0, 1, 1])
        self.obj.quality_bits["corrected"] = np.array([1, 0, 1, 1, 0, 0])
        expected = [0, 1, 0, 0, 0, 0]
        assert_array_almost_equal(self.obj._find_uncorrected_ice(), expected)

    def test_find_ice_above_rain(self):
        self.obj.is_ice = np.array([[1, 1, 0], [1, 0, 1]])
        self.obj.is_rain = np.array([1, 0])
        expected = [[1, 1, 0], [0, 0, 0]]
        assert_array_almost_equal(self.obj._find_ice_above_rain(), expected)

    def test_find_cold_above_rain(self):
        self.obj.category_bits["cold"] = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
        self.obj.category_bits["melting"] = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0]])
        self.obj.is_rain = np.array([1, 1, 0])
        expected = [[0, 0, 1], [0, 0, 1], [0, 0, 0]]
        assert_array_almost_equal(self.obj._find_cold_above_rain().data, expected)


class TestAppending:
    @pytest.fixture(autouse=True)
    def run_before_tests(self, categorize_file):
        self.ice_class = IceClassification(categorize_file)
        self.ier_source = IerSource(categorize_file, "ier")

    def test_append_ier_including_rain(self):
        self.ice_class.is_ice = np.array([[0, 1, 1], [1, 1, 0], [0, 0, 1]], dtype=bool)
        self.ier_source.append_main_variable_including_rain(self.ice_class)
        expected_mask = [[1, 0, 0], [0, 0, 1], [1, 1, 0]]
        assert_array_equal(self.ier_source.data["ier_inc_rain"][:].mask, expected_mask)

    def test_append_ier(self):
        self.ice_class.ice_above_rain = np.array([0, 0, 0, 1, 1], dtype=bool)
        self.ier_source.data["ier_inc_rain"] = np.ma.array([1, 2, 3, 4, 5], mask=[1, 0, 1, 0, 1])
        self.ier_source.append_main_variable(self.ice_class)
        expected_mask = [1, 0, 1, 1, 1]
        assert_array_equal(self.ier_source.data["ier"][:].mask, expected_mask)

    def test_append_ier_error(self):
        self.ier_source.data["ier_inc_rain"] = np.ma.array(
            [[1, 2], [3, 4], [5, 6]], mask=[[1, 0], [1, 0], [1, 0]]
        )
        self.ice_class.is_ice = np.array([[0, 0], [0, 1], [1, 1]], dtype=bool)
        self.ice_class.ice_above_rain = np.array([[1, 0], [1, 0], [1, 0]], dtype=bool)
        self.ier_source.append_ier_error(self.ice_class)
        expected_mask = [[1, 0], [1, 0], [1, 0]]
        assert_array_equal(self.ier_source.data["ier_error"][:].mask, expected_mask)
