import os
from tempfile import TemporaryDirectory

import pytest
from numpy.testing import assert_array_equal

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import disdrometer
from tests.unit.all_products_fun import Check

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

SITE_META = {
    "name": "Kumpula",
    "latitude": 50,
    "longitude": 104.5,
    "altitude": 50,
}


def test_format_time():
    assert disdrometer._format_thies_date("3.10.20") == "2020-10-03"


class TestParsivel(Check):
    date = "2021-03-18"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    site_meta = SITE_META
    filename = f"{SCRIPT_PATH}/data/parsivel/juelich.log"
    uuid = disdrometer.disdrometer2nc(filename, temp_path, site_meta)

    def test_global_attributes(self):
        assert "Parsivel" in self.nc.source
        assert self.nc.cloudnet_file_type == "disdrometer"
        assert self.nc.title == f'Parsivel2 disdrometer from {self.site_meta["name"]}'
        assert self.nc.year == "2021"
        assert self.nc.month == "03"
        assert self.nc.day == "18"
        assert self.nc.location == "Kumpula"

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size > 1000
        assert self.nc.dimensions["velocity"].size == 32
        assert self.nc.dimensions["diameter"].size == 32


class TestParsivel2(Check):
    date = "2019-11-09"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/norunda.log"
    site_meta = SITE_META
    uuid = disdrometer.disdrometer2nc(filename, temp_path, site_meta, date=date)

    def test_date_validation_fail(self, tmp_path):
        with pytest.raises(ValidTimeStampError):
            disdrometer.disdrometer2nc(
                self.filename,
                tmp_path / "invalid.nc",
                self.site_meta,
                date="2022-04-05",
            )


class TestParsivel3(Check):
    date = "2021-04-16"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/ny-alesund.log"
    site_meta = SITE_META
    uuid = disdrometer.disdrometer2nc(filename, temp_path, site_meta, date=date)


class TestThies(Check):
    date = "2021-09-15"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/thies-lnm/2021091507.txt"
    site_meta = SITE_META
    uuid = disdrometer.disdrometer2nc(filename, temp_path, site_meta, date=date)

    def test_processing(self):
        assert self.nc.title == f'LNM disdrometer from {self.site_meta["name"]}'
        assert self.nc.year == "2021"
        assert self.nc.month == "09"
        assert self.nc.day == "15"
        assert self.nc.location == "Kumpula"
        assert self.nc.cloudnet_file_type == "disdrometer"


class TestInvalidCharacters(Check):
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    filename = f"{SCRIPT_PATH}/data/parsivel/parsivel_bad.log"
    site_meta = SITE_META
    date = "2019-04-10"
    uuid = disdrometer.disdrometer2nc(filename, temp_path, site_meta, date=date)

    def test_masking(self):
        assert_array_equal(self.nc.variables["rainfall_rate"][:].mask, [0, 1, 0, 0, 0])
        assert_array_equal(self.nc.variables["n_particles"][:].mask, [0, 0, 0, 0, 0])
