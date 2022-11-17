import math
import sys
from os import path
from shutil import copytree
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import numpy.ma as ma
import pytest
from numpy.testing import assert_array_equal, assert_equal

from cloudnetpy.exceptions import InconsistentDataError, ValidTimeStampError
from cloudnetpy.instruments import rpg, rpg2nc

SCRIPT_PATH = path.dirname(path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from all_products_fun import SITE_META, Check
from radar_fun import RadarFun

FILEPATH = f"{SCRIPT_PATH}/data/rpg-fmcw-94"


class TestReduceHeader:
    n_points = 100
    header = {"a": n_points * [1], "b": n_points * [2], "c": n_points * [3]}

    def test_1(self):
        assert_equal(rpg._reduce_header(self.header), {"a": 1, "b": 2, "c": 3})

    def test_2(self):
        self.header["a"][50] = 10
        with pytest.raises(InconsistentDataError):
            assert_equal(rpg._reduce_header(self.header), {"a": 1, "b": 2, "c": 3})


class TestRPG2nc94GHz(Check):
    site_meta = SITE_META
    date = "2020-10-22"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/rpg.nc"
    uuid, valid_files = rpg2nc(FILEPATH, temp_path, site_meta, date=date)

    def test_variable_names(self):
        mandatory_variables = (
            "Zh",
            "v",
            "width",
            "ldr",
            "time",
            "range",
            "altitude",
            "latitude",
            "longitude",
            "radar_frequency",
            "nyquist_velocity",
            "zenith_angle",
            "skewness",
            "kurtosis",
            "rain_rate",
            "relative_humidity",
            "temperature",
            "pressure",
            "wind_speed",
            "wind_direction",
            "voltage",
            "brightness_temperature",
            "lwp",
            "if_power",
            "azimuth_angle",
            "status_flag",
            "transmitted_power",
            "transmitter_temperature",
            "receiver_temperature",
            "pc_temperature",
            "rho_cx",
            "phi_cx",
        )
        for key in mandatory_variables:
            assert key in self.nc.variables

    def test_long_names(self):
        data = [
            ("rho_cx", "Co-cross-channel correlation coefficient"),
            ("phi_cx", "Co-cross-channel differential phase"),
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].long_name
                assert value == expected, f"{value} != {expected}"

    def test_variables(self):
        assert math.isclose(self.nc.variables["radar_frequency"][:].data, 94.0, abs_tol=0.1)
        assert np.all(self.nc.variables["zenith_angle"][:].data) == 0

    def test_fill_values(self):
        bad_values = (-999, 1e-10)
        for key in self.nc.variables.keys():
            for value in bad_values:
                array = self.nc.variables[key][:]
                if array.ndim > 1:
                    assert not np.any(np.isclose(array, value)), f"{key} - {value}: {array}"

    def test_global_attributes(self):
        assert self.nc.source == "RPG-Radiometer Physics RPG-FMCW-94"
        assert self.nc.title == f'RPG-FMCW-94 cloud radar from {self.site_meta["name"]}'

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_default_processing(self, tmp_path):
        test_path = tmp_path / "default.nc"
        uuid, files = rpg2nc(FILEPATH, test_path, self.site_meta)
        assert len(files) == 3
        assert len(uuid) == 36

    def test_date_validation(self, tmp_path):
        test_path = tmp_path / "date.nc"
        uuid, files = rpg2nc(FILEPATH, test_path, self.site_meta, date=self.date)
        assert len(files) == 2

    def test_processing_of_one_file(self, tmp_path):
        test_path = tmp_path / "one.nc"
        uuid, files = rpg2nc(FILEPATH, test_path, self.site_meta, date="2020-10-23")
        assert len(files) == 1

    def test_incorrect_date_processing(self, tmp_path):
        test_path = tmp_path / "invalid.nc"
        with pytest.raises(ValidTimeStampError):
            rpg2nc(FILEPATH, test_path, self.site_meta, date="2010-10-24")

    def test_uuid_from_user(self, tmp_path):
        test_path = tmp_path / "uuid.nc"
        test_uuid = "abc"
        uuid, _ = rpg.rpg2nc(FILEPATH, test_path, self.site_meta, date="2020-10-23", uuid=test_uuid)
        assert uuid == test_uuid

    def test_handling_of_corrupted_files(self, tmp_path, tmp_path_factory):
        temp_dir = tmp_path_factory.mktemp("corrupt")
        test_path = tmp_path / "corrupt.nc"
        copytree(FILEPATH, temp_dir, dirs_exist_ok=True)
        (temp_dir / "foo.LV1").write_text("kissa")
        _, files = rpg.rpg2nc(str(temp_dir), test_path, self.site_meta, date="2020-10-22")
        assert len(files) == 2

    def test_geolocation_from_source_file(self, tmp_path):
        test_path = tmp_path / "geo.nc"
        meta_without_geolocation = {"name": "Kumpula", "altitude": 34}
        rpg.rpg2nc(FILEPATH, test_path, meta_without_geolocation)
        with netCDF4.Dataset(test_path) as nc:
            for key in ("latitude", "longitude"):
                assert key in nc.variables
                assert nc.variables[key][:] > 0


class TestRPG2ncSTSR35GHz(Check):
    site_meta = SITE_META
    date = "2021-09-13"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/rpg.nc"
    uuid, valid_files = rpg2nc(FILEPATH, temp_path, site_meta, date=date)

    def test_variable_names(self):
        mandatory_variables = (
            "Zh",
            "v",
            "width",
            "sldr",
            "time",
            "range",
            "altitude",
            "latitude",
            "longitude",
            "radar_frequency",
            "nyquist_velocity",
            "zenith_angle",
            "skewness",
            "kurtosis",
            "rain_rate",
            "relative_humidity",
            "temperature",
            "pressure",
            "wind_speed",
            "wind_direction",
            "voltage",
            "brightness_temperature",
            "lwp",
            "if_power",
            "azimuth_angle",
            "status_flag",
            "transmitted_power",
            "transmitter_temperature",
            "receiver_temperature",
            "pc_temperature",
            "zdr",
            "rho_hv",
            "phi_dp",
            "sldr",
            "srho_hv",
            "kdp",
            "differential_attenuation",
        )
        for key in mandatory_variables:
            assert key in self.nc.variables

    def test_long_names(self):
        data = [
            ("rho_cx", "Co-cross-channel correlation coefficient"),
            ("phi_cx", "Co-cross-channel differential phase"),
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].long_name
                assert value == expected, f"{value} != {expected}"

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_variables(self):
        assert math.isclose(self.nc.variables["radar_frequency"][:].data, 35.0, rel_tol=0.1)
        assert math.isclose(ma.median(self.nc.variables["zenith_angle"][:].data), 15, abs_tol=1)

    def test_fill_values(self):
        bad_values = (-999, 1e-10)
        for key in self.nc.variables.keys():
            for value in bad_values:
                array = self.nc.variables[key][:]
                if array.ndim > 1:
                    assert not np.any(np.isclose(array, value)), f"{key} - {value}: {array}"

    def test_global_attributes(self):
        assert self.nc.source == "RPG-Radiometer Physics RPG-FMCW-35"
        assert self.nc.title == f'RPG-FMCW-35 cloud radar from {self.site_meta["name"]}'
