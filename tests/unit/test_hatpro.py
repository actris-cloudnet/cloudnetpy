from os import path
from shutil import copytree
from tempfile import TemporaryDirectory

import netCDF4
import pytest
from numpy import ma

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import hatpro
from tests.unit.all_products_fun import Check

SCRIPT_PATH = path.dirname(path.realpath(__file__))
file_path = f"{SCRIPT_PATH}/data/hatpro-lwp/"


class TestHatpro2nc(Check):
    site_meta = {
        "name": "the_station",
        "altitude": 50,
        "latitude": 23.0,
        "longitude": 123,
    }
    date = "2020-07-23"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    uuid, valid_files = hatpro.hatpro2nc(file_path, temp_path, site_meta, date=date)

    def test_default_processing(self, tmp_path):
        test_path = tmp_path / "several.nc"
        uuid, files = hatpro.hatpro2nc(file_path, test_path, self.site_meta)
        assert len(files) == 4
        assert len(uuid) == 36

    def test_processing_of_several_lwp_files(self, tmp_path):
        test_uuid = "abc"
        test_path = tmp_path / "several.nc"
        uuid, files = hatpro.hatpro2nc(
            file_path,
            test_path,
            self.site_meta,
            date="2021-01-23",
            uuid=test_uuid,
        )
        assert len(files) == 2
        assert uuid == test_uuid
        with netCDF4.Dataset(test_path) as nc:
            time = nc.variables["time"]
            assert "hours since" in time.units
            assert max(time[:]) < 24
            for ind, t in enumerate(time[:-1]):
                assert time[ind + 1] > t
            assert "zenith_angle" in nc.variables
            assert "iwv" not in nc.variables
            assert "lwp" in nc.variables
            assert "kg m-2" in nc.variables["lwp"].units

    def test_processing_of_one_file(self, tmp_path):
        test_path = tmp_path / "one.nc"
        date = "2020-07-23"
        uuid, files = hatpro.hatpro2nc(file_path, test_path, self.site_meta, date=date)
        assert len(files) == 1

    def test_processing_of_no_files(self, tmp_path):
        test_path = tmp_path / "no.nc"
        with pytest.raises(ValidTimeStampError):
            hatpro.hatpro2nc(file_path, test_path, self.site_meta, date="2020-10-24")

    def test_handling_of_corrupted_files(self, tmp_path, tmp_path_factory):
        temp_dir = tmp_path_factory.mktemp("corrupt")
        copytree(file_path, temp_dir, dirs_exist_ok=True)
        (temp_dir / "foo.LV1").write_text("kissa")
        test_path = tmp_path / "corrupt.nc"
        _, files = hatpro.hatpro2nc(
            str(temp_dir),
            test_path,
            self.site_meta,
            date="2021-01-23",
        )
        assert len(files) == 2

    def test_lwp_and_iwv(self, tmp_path):
        file_path = f"{SCRIPT_PATH}/data/hatpro-lwp-iwv/"
        test_path = tmp_path / "full.nc"
        uuid, files = hatpro.hatpro2nc(file_path, test_path, self.site_meta)
        assert len(files) == 4
        assert len(uuid) == 36
        with netCDF4.Dataset(test_path) as nc:
            time = nc.variables["time"]
            assert "hours since" in time.units
            assert max(time[:]) < 24
            for ind, t in enumerate(time[:-1]):
                assert time[ind + 1] > t
            assert time.shape == (2036 + 1999,)
            assert "zenith_angle" in nc.variables
            assert "lwp" in nc.variables
            assert "kg m-2" in nc.variables["lwp"].units
            assert nc.variables["lwp"][0] is ma.masked
            assert ma.count_masked(nc.variables["lwp"][:]) == 1
            assert "iwv" in nc.variables
            assert "kg m-2" in nc.variables["iwv"].units
            assert nc.variables["iwv"][1] is ma.masked
            assert nc.variables["iwv"][-2] is ma.masked
            assert ma.count_masked(nc.variables["iwv"][:]) == 2

    def test_lwp_iwv_different_number_of_files(self, tmp_path):
        file_path = f"{SCRIPT_PATH}/data/hatpro-lwp-iwv-2/"
        test_path = tmp_path / "full.nc"
        uuid, files = hatpro.hatpro2nc(file_path, test_path, self.site_meta)
        assert len(files) == 3
        with netCDF4.Dataset(test_path) as nc:
            time = nc.variables["time"]
            assert "hours since" in time.units
            assert max(time[:]) < 24
            for ind, t in enumerate(time[:-1]):
                assert time[ind + 1] > t
            assert "zenith_angle" in nc.variables
            for key in ("lwp", "iwv"):
                assert key in nc.variables
            assert "kg m-2" in nc.variables["lwp"].units
            assert "kg m-2" in nc.variables["iwv"].units
            assert ma.count_masked(nc.variables["iwv"][:]) == 2034
            assert ma.count_masked(nc.variables["lwp"][:]) == 0

    def test_iwv_only(self, tmp_path):
        file_path = f"{SCRIPT_PATH}/data/hatpro-iwv/"
        test_path = tmp_path / "full.nc"
        with pytest.raises(ValidTimeStampError):
            hatpro.hatpro2nc(file_path, test_path, self.site_meta)
