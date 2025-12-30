import math
from os import path
from shutil import copytree
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_equal

from cloudnetpy.exceptions import InconsistentDataError, ValidTimeStampError
from cloudnetpy.instruments import rpg, rpg2nc
from tests.unit.all_products_fun import Check
from tests.unit.radar_fun import RadarFun

SCRIPT_PATH = path.dirname(path.realpath(__file__))
FILEPATH = f"{SCRIPT_PATH}/data/rpg-fmcw-94"


class TestRPG2nc94GHz(Check):
    site_meta = {
        "name": "Bucharest",
        "latitude": 44.344,
        "longitude": 26.012,
        "altitude": 77,
    }
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
            "rainfall_rate",
            "relative_humidity",
            "air_temperature",
            "air_pressure",
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
        assert math.isclose(
            self.nc.variables["radar_frequency"][:].data,
            94.0,
            abs_tol=0.1,
        )
        assert np.all(self.nc.variables["zenith_angle"][:].data) == 0

    def test_fill_values(self):
        bad_values = (-999, 1e-10)
        for key in self.nc.variables.keys():
            for value in bad_values:
                var = self.nc.variables[key]
                if var.dimensions == ("time", "range"):
                    array = var[:]
                    assert not np.any(
                        np.isclose(array, value),
                    ), f"{key} - {value}: {array}"

    def test_global_attributes(self):
        assert self.nc.source == "RPG-Radiometer Physics RPG-FMCW-94"
        assert self.nc.title == f"RPG-FMCW-94 cloud radar from {self.site_meta['name']}"

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_default_processing(self, tmp_path):
        test_path = tmp_path / "default.nc"
        with pytest.raises(ValueError):
            rpg2nc(FILEPATH, test_path, self.site_meta)

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
        test_uuid = "fe45561b-eb08-4d2a-a463-c6b4f7be7055"
        uuid, _ = rpg.rpg2nc(
            FILEPATH,
            test_path,
            self.site_meta,
            date="2020-10-23",
            uuid=test_uuid,
        )
        assert str(uuid) == test_uuid

    def test_handling_of_corrupted_files(self, tmp_path, tmp_path_factory):
        temp_dir = tmp_path_factory.mktemp("corrupt")
        test_path = tmp_path / "corrupt.nc"
        copytree(FILEPATH, temp_dir, dirs_exist_ok=True)
        (temp_dir / "foo.LV1").write_text("kissa")
        _, files = rpg.rpg2nc(
            str(temp_dir),
            test_path,
            self.site_meta,
            date="2020-10-22",
        )
        assert len(files) == 2

    def test_handling_of_corrupted_files_II(self, tmp_path, tmp_path_factory):
        temp_file = tmp_path / "temp.nc"
        filepath = f"{SCRIPT_PATH}/data/rpg-fmcw-94-corrupted"
        with pytest.raises(ValidTimeStampError):
            rpg2nc(filepath, str(temp_file), self.site_meta, date="2023-04-01")

    def test_geolocation_from_source_file(self, tmp_path):
        test_path = tmp_path / "geo.nc"
        meta_without_geolocation = {"name": "Kumpula", "altitude": 34}
        rpg.rpg2nc(FILEPATH, test_path, meta_without_geolocation, date="2020-10-22")
        with netCDF4.Dataset(test_path) as nc:
            for key in ("latitude", "longitude"):
                assert key in nc.variables
                assert np.all(nc.variables[key][:] > 0)


class TestRPG2ncDifferentChirps(Check):
    site_meta = {
        "name": "Norunda",
        "latitude": 60.086,
        "longitude": 17.481,
        "altitude": 15,
    }
    date = "2020-03-27"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/rpg.nc"
    uuid, valid_files = rpg2nc(
        f"{SCRIPT_PATH}/data/rpg-fmcw-94-chirps", temp_path, site_meta
    )

    def test_chirp_related_variables(self):
        time = self.nc.variables["time"][:]
        for key in self.nc.variables:
            var = self.nc.variables[key]
            if not var.dimensions == ("time", "chirp_sequence"):
                continue
            array = var[:]
            assert array.shape == (time.size, 4)
            assert not np.any(array[:, :3].mask)
            assert np.any(array[:, -1].mask)

    def test_nyquist_velocity(self):
        time = self.nc.variables["time"][:]
        range = self.nc.variables["range"][:]
        nyquist_velocity = self.nc.variables["nyquist_velocity"][:]
        assert nyquist_velocity.shape == (time.size, range.size)
        assert np.all(nyquist_velocity > 0)
        assert not np.any(nyquist_velocity.mask)


class TestRPG2ncSTSR35GHz(Check):
    site_meta = {
        "name": "Cabauw",
        "latitude": 51.968,
        "longitude": 4.927,
        "altitude": -1,
    }
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
            "rainfall_rate",
            "relative_humidity",
            "air_temperature",
            "air_pressure",
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
        assert math.isclose(
            self.nc.variables["radar_frequency"][:].data,
            35.0,
            rel_tol=0.1,
        )
        assert math.isclose(
            ma.median(self.nc.variables["zenith_angle"][:].data),
            15,
            abs_tol=1,
        )

    def test_fill_values(self):
        bad_values = (-999, 1e-10)
        for key in self.nc.variables.keys():
            for value in bad_values:
                var = self.nc.variables[key]
                if var.dimensions == ("time", "range"):
                    array = var[:]
                    assert not np.any(
                        np.isclose(array, value),
                    ), f"{key} - {value}: {array}"

    def test_global_attributes(self):
        assert self.nc.source == "RPG-Radiometer Physics RPG-FMCW-35"
        assert self.nc.title == f"RPG-FMCW-35 cloud radar from {self.site_meta['name']}"


@pytest.mark.parametrize(
    "data, expected",
    [
        ([ma.array([0, 0, 0], mask=[False, False, False]), ma.array([1, 1, 1])]),
        ([ma.array([0, 0, 0], mask=[False, False, True]), ma.array([1, 1, 0])]),
        ([ma.array([0, 0, 1], mask=[False, False, False]), ma.array([1, 1, 0])]),
        ([ma.array([16, 16, 16], mask=[False, False, False]), ma.array([0, 0, 0])]),
        ([ma.array([-4, -4, -4], mask=[False, False, False]), ma.array([1, 1, 1])]),
        ([ma.array([-6, -6, -6], mask=[False, False, False]), ma.array([0, 0, 0])]),
        ([ma.array([-34, 233, 21214], mask=[True, True, True]), ma.array([0, 0, 0])]),
        (
            [
                ma.array([0, 0, 50, 50, 50], mask=[False, False, False, False, False]),
                ma.array([1, 1, 0, 0, 0]),
            ]
        ),
    ],
)
def test_filter_zenith_angle(data: ma.MaskedArray, expected: ma.MaskedArray):
    res = rpg._filter_zenith_angle(data)
    assert_equal(res.data, expected.data)
    if isinstance(res, ma.MaskedArray):
        assert_equal(res.mask, expected.mask)
