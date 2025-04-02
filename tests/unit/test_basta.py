from os import path
from tempfile import TemporaryDirectory

import netCDF4
import pytest

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import basta2nc
from tests.unit.all_products_fun import Check
from tests.unit.radar_fun import RadarFun

SCRIPT_PATH = path.dirname(path.realpath(__file__))
filename = f"{SCRIPT_PATH}/data/basta/basta_1a_cldradLz1R025m_v03_20210827_000000.nc"


class TestBASTA(Check):
    site_meta = {
        "name": "Palaiseau",
        "latitude": 48.717,
        "longitude": 2.209,
        "altitude": 158,
    }
    date = "2021-08-27"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    uuid = basta2nc(filename, temp_path, site_meta)

    def test_variable_names(self):
        keys = {
            "Zh",
            "v",
            "time",
            "range",
            "radar_frequency",
            "height",
            "nyquist_velocity",
            "latitude",
            "longitude",
            "altitude",
            "zenith_angle",
            "radar_pitch",
            "radar_yaw",
            "radar_roll",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_variables(self):
        assert self.nc.variables["radar_frequency"][:].data == 95.0  # Hard coded

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_geolocation_from_source_file(self, tmp_path):
        test_path = tmp_path / "geo.nc"
        meta_without_geolocation = {"name": "Kumpula"}
        basta2nc(filename, test_path, meta_without_geolocation)
        with netCDF4.Dataset(test_path) as nc:
            for key in ("latitude", "longitude", "altitude"):
                assert key in nc.variables
                assert nc.variables[key][:] > 0

    def test_global_attributes(self):
        assert self.nc.source == "BASTA"
        assert self.nc.title == "BASTA cloud radar from Palaiseau"

    def test_wrong_date_validation(self, tmp_path):
        with pytest.raises(ValidTimeStampError):
            basta2nc(
                filename,
                tmp_path / "invalid.nc",
                self.site_meta,
                date="2021-01-04",
            )

    def test_uuid_from_user(self, tmp_path):
        test_path = tmp_path / "uuid.nc"
        uuid_from_user = "kissa"
        uuid = basta2nc(filename, test_path, self.site_meta, uuid=uuid_from_user)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.file_uuid == uuid_from_user
            assert uuid == uuid_from_user
