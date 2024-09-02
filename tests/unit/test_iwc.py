import netCDF4
import numpy as np
import pytest
from numpy import ma, testing
from numpy.testing import assert_array_equal

from cloudnetpy.products.iwc import IceClassification, IwcSource


@pytest.fixture(scope="session")
def categorize_file(tmpdir_factory, file_metadata):
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    with netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC") as nc:
        dimensions = {
            "time": 3,
            "height": 2,
            "model_time": 3,
            "model_height": 2,
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
        nc.createVariable("Z_bias", "f8")[:] = 0.5
        nc.createVariable("radar_frequency", "f8")[:] = 35.5
        nc.createVariable("Z", "f8", ("time", "height"))[:] = np.array(
            [[10, 20], [10, 20], [10, 20]],
        )
        nc.createVariable("Z_error", "f8", ("time", "height"))[:] = np.array(
            [[1, 2], [1, 2], [2, 3]],
        )
        nc.createVariable("category_bits", "i4", ("time", "height"))[:] = np.array(
            [[0, 1], [2, 3], [4, 8]],
        )
        nc.createVariable("quality_bits", "i4", ("time", "height"))[:] = np.array(
            [[0, 1], [2, 3], [4, 8]],
        )
        temperature = np.array([[280, 290], [280, 290], [280, 290]])
        nc.createVariable("temperature", "f8", ("model_time", "model_height"))[
            :
        ] = temperature
        nc.createVariable("Z_sensitivity", "f8", "height")[:] = 2.0
        nc.createVariable("rainfall_rate", "i4", "time")[:] = [0, 1, 0]
    return file_name


def test_iwc_wl_band(categorize_file):
    obj = IwcSource(categorize_file, "iwc")
    assert obj.wl_band == 0


@pytest.mark.parametrize("result", ["K2liquid0", "ZT", "T", "Z", "c"])
def test_iwc_coeffs(result, categorize_file):
    obj = IwcSource(categorize_file, "iwc")
    assert result in obj.coefficients._fields
    assert obj.coefficients == (0.878, 0.000242, -0.0186, 0.0699, -1.63)


def test_iwc_temperature(categorize_file):
    obj = IwcSource(categorize_file, "iwc")
    expected = [[6.85, 16.85], [6.85, 16.85], [6.85, 16.85]]
    testing.assert_almost_equal(obj.temperature, expected)


class TestIceClassification:
    @pytest.fixture(autouse=True)
    def run_before_tests(self, categorize_file):
        self.obj = IceClassification(categorize_file)

    def test_find_ice(self):
        self.obj.category_bits.falling = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1])
        self.obj.category_bits.freezing = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
        self.obj.category_bits.melting = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.obj.category_bits.insect = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0])
        expected = [0, 1, 1, 0, 1, 0, 0, 1, 0]
        testing.assert_array_equal(self.obj._find_ice(), expected)

    def test_would_be_ice(self):
        self.obj.category_bits.falling = np.array([1, 1, 1, 0, 1, 1, 0, 1, 1])
        self.obj.category_bits.freezing = np.array([0, 0, 1, 0, 0, 1, 0, 0, 1])
        self.obj.category_bits.melting = np.array([0, 0, 0, 1, 0, 0, 0, 0, 1])
        self.obj.category_bits.insect = np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])
        expected = [0, 1, 0, 1, 1, 0, 0, 1, 1]
        testing.assert_array_equal(self.obj._find_would_be_ice(), expected)

    def test_find_corrected_ice(self):
        self.obj.is_ice = np.array([1, 1, 1, 1, 0, 0])
        self.obj.quality_bits.attenuated_liquid = np.array([1, 1, 1, 0, 1, 0])
        self.obj.quality_bits.corrected_liquid = np.array([1, 0, 0, 1, 1, 0])
        self.obj.quality_bits.attenuated_rain = np.array([0, 0, 0, 0, 0, 0])
        self.obj.quality_bits.corrected_rain = np.array([0, 0, 0, 0, 0, 0])
        self.obj.quality_bits.attenuated_melting = np.array([0, 0, 0, 0, 0, 0])
        self.obj.quality_bits.corrected_melting = np.array([0, 0, 0, 0, 0, 0])
        self.obj._is_attenuated = self.obj._find_attenuated()
        self.obj._is_corrected = self.obj._find_corrected()
        expected = [1, 0, 0, 0, 0, 0]
        testing.assert_array_equal(self.obj._find_corrected_ice(), expected)

    # def test_find_uncorrected_ice(self):
    #     self.obj.is_ice = np.array([1, 1, 1, 1, 0, 0])
    #     self.obj.quality_bits.attenuated_liquid = np.array([1, 1, 0, 0, 1, 1])
    #     self.obj.quality_bits.corrected_liquid = np.array([1, 0, 1, 1, 0, 0])
    #     self.obj.quality_bits.attenuated_rain = np.array([0, 0, 0, 0, 0, 0])
    #     self.obj.quality_bits.corrected_rain = np.array([0, 0, 0, 0, 0, 0])
    #     self.obj.quality_bits.attenuated_melting = np.array([0, 0, 0, 0, 0, 0])
    #     self.obj.quality_bits.corrected_melting = np.array([0, 0, 0, 0, 0, 0])
    #     expected = [0, 1, 0, 0, 0, 0]
    #     testing.assert_array_equal(self.obj._find_uncorrected_ice(), expected)

    def test_find_ice_above_rain(self):
        self.obj.is_ice = np.array([[1, 1, 0], [1, 0, 1]])
        self.obj.is_rain = np.array([1, 0])
        expected = [[1, 1, 0], [0, 0, 0]]
        testing.assert_array_equal(self.obj._find_ice_above_rain(), expected)

    # def test_find_cold_above_rain(self):
    #     self.obj.category_bits.freezing = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]])
    #     self.obj.category_bits.melting = np.array([[0, 1, 0], [1, 1, 0], [1, 0, 0]])
    #     self.obj.is_rain = np.array([1, 1, 0])
    #     expected = [[0, 0, 1], [0, 0, 1], [0, 0, 0]]
    #     testing.assert_array_equal(self.obj._find_cold_above_rain().data, expected)


class TestAppending:
    @pytest.fixture(autouse=True)
    def run_before_tests(self, categorize_file):
        self.ice_class = IceClassification(categorize_file)
        self.iwc_source = IwcSource(categorize_file, "iwc")

    # def test_append_iwc(self):
    #     self.ice_class.ice_above_rain = np.array([0, 0, 0, 1, 1], dtype=bool)
    #     self.iwc_source.data["iwc"] = ma.array(
    #         [1, 2, 3, 4, 5],
    #         mask=[1, 0, 1, 0, 1],
    #     )
    #     self.iwc_source.append_icy_data(self.ice_class)
    #     expected_mask = [1, 0, 1, 1, 1]
    #     assert_array_equal(self.iwc_source.data["iwc"][:].mask, expected_mask)

    # def test_iwc_error(self):
    #     self.ice_class.is_ice = np.array([[0, 0], [0, 1], [1, 1]], dtype=bool)
    #     self.ice_class.ice_above_rain = np.array([[1, 0], [1, 0], [1, 0]], dtype=bool)
    #     self.iwc_source.append_error(self.ice_class)
    #     expected_mask = [[1, 1], [1, 0], [1, 0]]
    #     assert_array_equal(self.iwc_source.data["iwc_error"][:].mask, expected_mask)

    def test_append_sensitivity(self):
        self.iwc_source.append_sensitivity()
        assert self.iwc_source.data["iwc_sensitivity"][:].shape == (2,)

    def test_append_bias(self):
        self.iwc_source.append_bias()
        assert isinstance(self.iwc_source.data["iwc_bias"].data[()], float)


# TODO: Fix this test
# def test_append_iwc_status(categorize_file):
#     iwc_source = IwcSource(categorize_file, "iwc")
#     ice_class = IceClassification(categorize_file)
#     iwc_source.data["iwc"] = ma.array(
#         [[1, 1], [1, 1], [1, 1]],
#         dtype=float,
#         mask=[[1, 0], [0, 0], [0, 0]],
#     )
#     ice_class.is_ice = np.array([[1, 0], [0, 0], [0, 0]], dtype=bool)
#     ice_class.corrected_ice = np.array([[0, 0], [1, 0], [0, 1]], dtype=bool)
#     ice_class.uncorrected_ice = np.array([[0, 0], [0, 1], [1, 0]], dtype=bool)
#     #ice_class.cold_above_rain = np.array([[0, 0], [0, 0], [1, 0]], dtype=bool)
#     ice_class.ice_above_rain = np.array([[0, 0], [0, 0], [0, 1]], dtype=bool)
#     ice_class.would_be_ice = np.array([[0, 0], [0, 0], [0, 0]], dtype=bool)
#     iwc_source.append_status(ice_class)
#     for value in range(1, 7):
#         assert value in iwc_source.data["iwc_retrieval_status"][:]
