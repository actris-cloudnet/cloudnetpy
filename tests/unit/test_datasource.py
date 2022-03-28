from numpy.testing import assert_array_equal
from cloudnetpy.categorize import datasource
import pytest


class TestDataSource:
    @pytest.fixture(autouse=True)
    def init_tests(self, nc_file):
        self.obj = datasource.DataSource(nc_file)

    def test_init_altitude(self, file_metadata):
        assert self.obj.altitude == file_metadata["altitude_km"] * 1000

    def test_getvar(self, test_array):
        assert_array_equal(self.obj.getvar("model_height"), test_array)

    def test_get_date(self):
        assert_array_equal(self.obj.get_date(), ["2019", "05", "23"])

    def test_getvar_missing(self):
        with pytest.raises(RuntimeError):
            self.obj.getvar("not_existing_variable")

    def test_init_time(self, test_array):
        assert_array_equal(self.obj.time, test_array)

    def test_close(self):
        assert self.obj.dataset.isopen() is True
        self.obj.close()
        assert self.obj.dataset.isopen() is False

    def test_km2m(self, test_array):
        alt = self.obj.dataset.variables["range"]
        assert_array_equal(self.obj.km2m(alt), test_array * 1000)

    def test_m2km(self, test_array):
        alt = self.obj.dataset.variables["height"]
        assert_array_equal(self.obj.m2km(alt), test_array / 1000)

    def test_get_height_m(self, test_array):
        assert_array_equal(self.obj.height, test_array)
