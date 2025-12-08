from os import path
from tempfile import TemporaryDirectory

import pytest
from numpy.testing import assert_allclose

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import radiometrics2nc
from tests.unit.all_products_fun import Check

SCRIPT_PATH = path.dirname(path.realpath(__file__))


SITE_META = {
    "name": "the_station",
    "altitude": 50,
    "latitude": 23.0,
    "longitude": 123,
}


class TestRadiometrics2nc(Check):
    test_input = f"{SCRIPT_PATH}/data/radiometrics/2021-07-18_00-00-00_lv2.csv"
    date = "2021-07-18"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/radiometrics.nc"
    site_meta = SITE_META
    uuid = radiometrics2nc(test_input, temp_path, site_meta, date=date)

    def test_time(self):
        assert_allclose(self.nc.variables["time"][:], [0.02, 0.10527778])

    def test_range(self):
        assert len(self.nc.variables["range"]) == 47
        assert_allclose(self.nc.variables["range"][:5], [0, 100, 200, 300, 400])
        assert_allclose(
            self.nc.variables["range"][-5:], [9000, 9250, 9500, 9750, 10000]
        )

    def test_lwp(self):
        assert_allclose(self.nc.variables["lwp"][:], [0.030, 0.030])

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
                self.test_input,
                test_path,
                SITE_META,
                date="2021-07-19",
            )


class TestRadiometrics2ncAgain(Check):
    test_input = f"{SCRIPT_PATH}/data/radiometrics/2021-10-06_00-04-08_lv2.csv"
    site_meta = SITE_META
    date = "2021-10-06"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/radiometrics.nc"
    uuid = radiometrics2nc(test_input, temp_path, site_meta, date=date)

    def test_time(self):
        assert_allclose(
            self.nc.variables["time"][:],
            [0.08305556, 0.11055555, 0.13833334, 0.16611111],
        )

    def test_range(self):
        assert len(self.nc.variables["range"]) == 58
        assert_allclose(self.nc.variables["range"][:5], [0, 50, 100, 150, 200])
        assert_allclose(
            self.nc.variables["range"][-5:], [9000, 9250, 9500, 9750, 10000]
        )

    def test_lwp(self):
        assert_allclose(self.nc.variables["lwp"][:], [0.164, 0.198, 0.161, 0.202])

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
                self.test_input,
                test_path,
                self.site_meta,
                date="2021-10-07",
            )


class TestRadiometrics2ncSkipNonZenith(Check):
    test_input = f"{SCRIPT_PATH}/data/radiometrics/2024-01-22_00-04-09_lv2.csv"
    site_meta = SITE_META
    date = "2024-01-22"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/radiometrics.nc"
    uuid = radiometrics2nc(test_input, temp_path, site_meta, date=date)

    def test_time(self):
        assert_allclose(self.nc.variables["time"][:], [5 / 60 + 23 / 60 / 60])

    def test_lwp(self):
        assert_allclose(self.nc.variables["lwp"][:], [0.007])

    def test_iwv(self):
        assert_allclose(self.nc.variables["iwv"][:], [12.64])


class TestRadiometrics2ncWVR(Check):
    test_input = f"{SCRIPT_PATH}/data/radiometrics/20140106_1126.los"
    site_meta = SITE_META
    date = "2014-01-06"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/radiometrics.nc"
    uuid = radiometrics2nc(test_input, temp_path, site_meta, date=date)

    def test_time(self):
        assert_allclose(
            self.nc.variables["time"][:],
            [
                11 + 28 / 60 + 7 / 60 / 60,
                11 + 29 / 60 + 35 / 60 / 60,
                11 + 30 / 60 + 3 / 60 / 60,
                11 + 30 / 60 + 32 / 60 / 60,
                11 + 31 / 60 + 1 / 60 / 60,
            ],
        )

    def test_lwp(self):
        assert_allclose(self.nc.variables["lwp"][:], [0.0, 0.0, 0.0, 0.0, 3.640])

    def test_iwv(self):
        assert_allclose(self.nc.variables["iwv"][:], [23.07, 22.50, 22.92, 22.51, 0.0])

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
                self.test_input,
                test_path,
                self.site_meta,
                date="2014-01-07",
            )


class TestRadiometrics2ncMissing(Check):
    test_input = f"{SCRIPT_PATH}/data/radiometrics/20100926_0005.los"
    site_meta = SITE_META
    date = "2010-09-26"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/radiometrics.nc"
    uuid = radiometrics2nc(test_input, temp_path, site_meta, date=date)

    def test_time(self):
        assert_allclose(
            self.nc.variables["time"][:],
            [
                6 / 60 + 18 / 60 / 60,
                7 / 60 + 44 / 60 / 60,
            ],
        )

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
                self.test_input,
                test_path,
                self.site_meta,
                date="2014-01-07",
            )


class TestRadiometrics2ncInvalid(Check):
    test_input = f"{SCRIPT_PATH}/data/radiometrics/20131220_1319.los"
    site_meta = SITE_META
    date = "2013-12-20"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/radiometrics.nc"
    uuid = radiometrics2nc(test_input, temp_path, site_meta, date=date)

    def test_time(self):
        assert_allclose(
            self.nc.variables["time"][:],
            [
                23 + 16 / 60 + 31 / 60 / 60,
                23 + 17 / 60 + 29 / 60 / 60,
            ],
        )

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
                self.test_input,
                test_path,
                self.site_meta,
                date="2014-01-07",
            )


class TestOldDWDMP3039AFileFormat(Check):
    test_input = f"{SCRIPT_PATH}/data/radiometrics/2010-10-01_00-00-09_lv2.csv"
    date = "2010-10-01"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/radiometrics.nc"
    site_meta = SITE_META
    uuid = radiometrics2nc(test_input, temp_path, site_meta, date=date)

    def test_default_processing(self, tmp_path):
        test_path = tmp_path / "default.nc"
        radiometrics2nc(self.test_input, test_path, self.site_meta)

    def test_time(self):
        time = self.nc.variables["time"][:]
        assert len(time) == 4

    def test_masking(self):
        temp = self.nc.variables["temperature"][:]
        assert temp.mask.sum() == 1


class TestRadiometricsAlternatingProcWhere301before401(Check):
    test_input = f"{SCRIPT_PATH}/data/radiometrics/2025-09-30_00-00-45_lv2.csv"
    date = "2025-09-30"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/radiometrics.nc"
    site_meta = SITE_META
    uuid = radiometrics2nc(test_input, temp_path, site_meta, date=date)

    def test_time(self):
        time = self.nc.variables["time"][:]
        assert_allclose(time, [4 / 60 + 18 / 60 / 60])

    def test_lwp(self):
        lwp = self.nc.variables["lwp"][:]
        assert_allclose(lwp, [0])

    def test_iwv(self):
        iwv = self.nc.variables["iwv"][:]
        assert_allclose(iwv, [11.14])
