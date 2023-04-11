import sys
from os import path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest

from cloudnetpy.exceptions import InconsistentDataError, ValidTimeStampError
from cloudnetpy.instruments import mira
from tests.unit.all_products_fun import Check

SCRIPT_PATH = path.dirname(path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from tests.unit.radar_fun import RadarFun

filepath = f"{SCRIPT_PATH}/data/mira/"


class TestMeasurementDate:
    correct_date = ["2020", "05", "24"]

    @pytest.fixture(autouse=True)
    def _init(self, raw_mira_file):
        self.raw_radar = mira.Mira(raw_mira_file, {"name": "Test"})

    def test_validate_date(self):
        self.raw_radar.screen_by_date("2020-05-24")
        assert self.raw_radar.date == self.correct_date

    def test_validate_date_fails(self):
        with pytest.raises(ValidTimeStampError):
            self.raw_radar.screen_by_date("2020-05-23")


class TestMIRA2nc(Check):
    site_meta = {
        "name": "Punta Arenas",
        "latitude": 50,
        "longitude": 104.5,
        "altitude": 50,
    }
    date = "2021-01-02"
    n_time1 = 146
    n_time2 = 145
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/mira.nc"
    uuid = mira.mira2nc(f"{filepath}/20210102_0000.mmclx", temp_path, site_meta)

    def test_variable_names(self):
        keys = {
            "Zh",
            "v",
            "width",
            "ldr",
            "SNR",
            "time",
            "range",
            "radar_frequency",
            "nyquist_velocity",
            "latitude",
            "longitude",
            "altitude",
            "zenith_angle",
            "azimuth_angle",
            "height",
            "rg0",
            "nave",
            "prf",
            "nfft",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_variables(self):
        assert self.nc.variables["radar_frequency"][:].data == 35.5  # Hard coded
        assert np.all(self.nc.variables["zenith_angle"][:].data) == 0

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_long_names(self):
        data = [
            ("nfft", "Number of FFT points"),
            (
                "nave",
                "Number of spectral averages (not accounting for overlapping FFTs)",
            ),
            ("rg0", "Number of lowest range gates"),
            ("prf", "Pulse Repetition Frequency"),
            ("SNR", "Signal-to-noise ratio"),
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].long_name
                assert value == expected, f"{value} != {expected}"

    def test_processing_of_one_nc_file(self):
        assert len(self.nc.variables["time"][:]) == self.n_time1

    def test_global_attributes(self):
        assert self.nc.source == "METEK MIRA-35"
        assert self.nc.title == f'MIRA-35 cloud radar from {self.site_meta["name"]}'

    def test_processing_of_several_nc_files(self, tmp_path):
        test_path = tmp_path / "several.nc"
        mira.mira2nc(filepath, test_path, self.site_meta)
        with netCDF4.Dataset(test_path) as nc:
            assert len(nc.variables["time"][:]) == self.n_time1 + self.n_time2

    def test_correct_date_validation(self, tmp_path):
        test_path = tmp_path / "date.nc"
        mira.mira2nc(
            f"{filepath}/20210102_0000.mmclx",
            test_path,
            self.site_meta,
            date="2021-01-02",
        )

    def test_wrong_date_validation(self, tmp_path):
        test_path = tmp_path / "invalid.nc"
        with pytest.raises(ValidTimeStampError):
            mira.mira2nc(
                f"{filepath}/20210102_0000.mmclx",
                test_path,
                self.site_meta,
                date="2021-01-03",
            )

    def test_uuid_from_user(self, tmp_path):
        test_path = tmp_path / "uuid.nc"
        uuid_from_user = "kissa"
        uuid = mira.mira2nc(
            f"{filepath}/20210102_0000.mmclx",
            test_path,
            self.site_meta,
            uuid=uuid_from_user,
        )
        with netCDF4.Dataset(test_path) as nc:
            assert nc.file_uuid == uuid_from_user
            assert uuid == uuid_from_user

    def test_geolocation_from_source_file(self, tmp_path):
        test_path = tmp_path / "geo.nc"
        meta_without_geolocation = {"name": "Kumpula"}
        mira.mira2nc(
            f"{filepath}/20210102_0000.mmclx",
            test_path,
            meta_without_geolocation,
        )
        with netCDF4.Dataset(test_path) as nc:
            for key in ("latitude", "longitude", "altitude"):
                assert key in nc.variables
                assert nc.variables[key][:] > 0


def test_allow_vary_option():
    site_meta = {
        "name": "Juelich",
        "latitude": 50,
        "longitude": 104.5,
        "altitude": 50,
    }
    temp_dir = TemporaryDirectory()
    temp_path = f"{temp_dir.name}/mira.nc"
    date = "2021-11-24"
    filepath = f"{SCRIPT_PATH}/data/mira_inconsistent/"
    _ = mira.mira2nc(filepath, temp_path, site_meta=site_meta, date=date)
