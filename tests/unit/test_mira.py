import datetime
from os import path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest

from cloudnetpy.exceptions import RadarDataError, ValidTimeStampError
from cloudnetpy.instruments import mira
from tests.unit.all_products_fun import Check
from tests.unit.radar_fun import RadarFun

SCRIPT_PATH = path.dirname(path.realpath(__file__))
filepath = f"{SCRIPT_PATH}/data/mira/"

SITE_META = {
    "name": "Punta Arenas",
    "latitude": 50,
    "longitude": 104.5,
    "altitude": 50,
}


class TestMeasurementDate:
    correct_date = datetime.date(2020, 5, 24)

    @pytest.fixture(autouse=True)
    def _init(self, raw_mira_file):
        self.raw_radar = mira.Mira(raw_mira_file, {"name": "Test"})

    def test_validate_date(self):
        self.raw_radar.screen_by_date(datetime.date(2020, 5, 24))
        assert self.raw_radar.date == self.correct_date

    def test_validate_date_fails(self):
        with pytest.raises(ValidTimeStampError):
            self.raw_radar.screen_by_date(datetime.date(2020, 5, 23))


class TestMIRA2nc(Check):
    site_meta = SITE_META
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
            "snr_limit",
            "tpow",
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
        assert self.nc.title == f"MIRA-35 cloud radar from {self.site_meta['name']}"

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
        uuid_from_user = "fe45561b-eb08-4d2a-a463-c6b4f7be7055"
        uuid = mira.mira2nc(
            f"{filepath}/20210102_0000.mmclx",
            test_path,
            self.site_meta,
            uuid=uuid_from_user,
        )
        with netCDF4.Dataset(test_path) as nc:
            assert nc.file_uuid == uuid_from_user
            assert str(uuid) == uuid_from_user

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
                assert np.all(nc.variables[key][:] > 0)


def test_allow_vary_option():
    temp_dir = TemporaryDirectory()
    temp_path = f"{temp_dir.name}/mira.nc"
    date = "2021-11-24"
    filepath = f"{SCRIPT_PATH}/data/mira_inconsistent/"
    _ = mira.mira2nc(filepath, temp_path, site_meta=SITE_META, date=date)


class TestZncFiles(Check):
    site_meta = SITE_META
    date = "2023-02-01"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/mira.nc"
    filepath = f"{SCRIPT_PATH}/data/mira_znc/"
    uuid = mira.mira2nc(f"{filepath}20230201_0900_mbr5-trunc.znc", temp_path, site_meta)

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_wrong_path(self):
        filepath = f"{SCRIPT_PATH}/data/vaisala/"
        with pytest.raises(FileNotFoundError):
            mira.mira2nc(
                filepath,
                self.temp_path,
                site_meta=self.site_meta,
                date=self.date,
            )

    def test_if_both_types_of_files(self):
        files = [
            f"{self.filepath}20230201_0900_mbr5-trunc.znc",
            f"{self.filepath}20230201_0900_mbr5-trunc.mmclx",
        ]
        with pytest.raises(TypeError):
            mira.mira2nc(
                files,
                self.temp_path,
                site_meta=self.site_meta,
                date=self.date,
            )

    def test_wrong_kind_of_file(self):
        input_file = f"{SCRIPT_PATH}/data/chm15k/00100_A202010222015_CHM170137.nc"
        with pytest.raises(ValueError):
            mira.mira2nc(
                input_file,
                self.temp_path,
                site_meta=self.site_meta,
                date=self.date,
            )

    def test_list_of_wrong_kind_of_files(self):
        filepath = f"{SCRIPT_PATH}/data/chm15k/"
        files = [
            f"{filepath}00100_A202010222015_CHM170137.nc",
            f"{filepath}00100_A202010220005_CHM170137.nc",
        ]
        with pytest.raises(ValueError):
            mira.mira2nc(
                files,
                self.temp_path,
                site_meta=self.site_meta,
                date=self.date,
            )

    def test_takes_znc_as_default(self):
        filepath = f"{SCRIPT_PATH}/data/mira_znc/"
        temp_dir = TemporaryDirectory()
        temp_path = f"{temp_dir.name}/mira.nc"
        mira.mira2nc(filepath, temp_path, site_meta=self.site_meta, date=self.date)
        with netCDF4.Dataset(temp_path) as nc:
            assert len(nc.variables["time"][:]) == 5


class TestSTSRFiles(Check):
    site_meta = SITE_META
    date = "2023-02-01"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/mira.nc"
    filepath = f"{SCRIPT_PATH}/data/mira_stsr/"
    uuid = mira.mira2nc(
        f"{filepath}20230201_0900_mbr7_stsr-trunc.znc",
        temp_path,
        site_meta,
    )

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()


class TestFilesHavingNyquistVelocityVector(Check):
    site_meta = SITE_META
    date = "2020-01-16"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/mira.nc"
    filepath = f"{SCRIPT_PATH}/data/mira-nyquist/"
    uuid = mira.mira2nc(f"{filepath}20200116_0000-trunc.mmclx", temp_path, SITE_META)

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()


def test_masked_mira():
    filepath = f"{SCRIPT_PATH}/data/mira-masked/"
    with pytest.raises(ValidTimeStampError):
        mira.mira2nc(f"{filepath}", "temp.nc", SITE_META)


@pytest.mark.parametrize(
    "filename, suffix",
    [
        ("20251013_120000.azisectorscan.mmclx", "mmclx"),
        ("20251013_120000.dmp", "dmp"),
        ("20251013_120000.man.mmclx", "mmclx"),
        ("20251013_120000.man.mmclx.01", "mmclx"),
        ("20251013_120000.mmclx", "mmclx"),
        ("20251013_120000.mmclx.01", "mmclx"),
        ("20251013_120000.ppibb.mmclx", "mmclx"),
        ("20251013_120000.ppi.mmclx", "mmclx"),
        ("20251013_120000.rhi.mmclx", "mmclx"),
        ("20251013_120000.rhistep.mmclx", "mmclx"),
        ("20251013_120000.vert.mmclx", "mmclx"),
        ("20251013_120000.zen.mmclx", "mmclx"),
        ("20251013_120000.znc", "znc"),
        ("20251013_120000", ""),
    ],
)
def test_get_suffix(filename: str, suffix: str):
    assert mira._get_suffix(filename) == suffix
