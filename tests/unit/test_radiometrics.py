from os import path
from tempfile import TemporaryDirectory

import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import radiometrics2nc
from tests.unit.all_products_fun import Check

SCRIPT_PATH = path.dirname(path.realpath(__file__))


class TestRadiometrics2nc(Check):
    test_input = f"{SCRIPT_PATH}/data/radiometrics/2021-07-18_00-00-00_lv2.csv"
    site_meta = {
        "name": "the_station",
        "altitude": 50,
        "latitude": 23.0,
        "longitude": 123,
    }
    date = "2021-07-18"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/radiometrics.nc"
    uuid = radiometrics2nc(test_input, temp_path, site_meta, date=date)

    def test_time(self):
        assert_allclose(self.nc.variables["time"][:], [0.02, 0.10527778, 0.1886111])

    def test_lwp(self):
        assert_array_equal(self.nc.variables["lwp"][:], [30, 30, 0])

    def test_global_attributes(self):
        assert self.nc.source == "Radiometrics"
        assert self.nc.title == "Microwave radiometer from the_station"

    def test_default_processing(self, tmp_path):
        test_path = tmp_path / "default.nc"
        radiometrics2nc(self.test_input, test_path, self.site_meta)

    def test_processing_of_one_file(self, tmp_path):
        test_path = tmp_path / "one.nc"
        radiometrics2nc(self.test_input, test_path, self.site_meta, date=self.date)

    def test_processing_of_no_files(self, tmp_path):
        test_path = tmp_path / "no.nc"
        with pytest.raises(ValidTimeStampError):
            radiometrics2nc(
                self.test_input, test_path, self.site_meta, date="2021-07-19"
            )


class TestRadiometrics2ncAgain(Check):
    test_input = f"{SCRIPT_PATH}/data/radiometrics/2021-10-06_00-04-08_lv2.csv"
    site_meta = {
        "name": "the_station",
        "altitude": 50,
        "latitude": 23.0,
        "longitude": 123,
    }
    date = "2021-10-06"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/radiometrics.nc"
    uuid = radiometrics2nc(test_input, temp_path, site_meta, date=date)

    def test_time(self):
        assert_allclose(
            self.nc.variables["time"][:],
            [0.08305556, 0.11055555, 0.13833334, 0.16611111],
        )

    def test_lwp(self):
        assert_array_equal(self.nc.variables["lwp"][:], [164, 198, 161, 202])

    def test_default_processing(self, tmp_path):
        test_path = tmp_path / "default.nc"
        radiometrics2nc(self.test_input, test_path, self.site_meta)

    def test_processing_of_one_file(self, tmp_path):
        test_path = tmp_path / "one.nc"
        radiometrics2nc(self.test_input, test_path, self.site_meta, date=self.date)

    def test_processing_of_no_files(self, tmp_path):
        test_path = tmp_path / "no.nc"
        with pytest.raises(ValidTimeStampError):
            radiometrics2nc(
                self.test_input, test_path, self.site_meta, date="2021-10-07"
            )
