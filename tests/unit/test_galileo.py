from os import path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest

from cloudnetpy.exceptions import ValidTimeStampError, RadarDataError
from cloudnetpy.instruments import galileo
from tests.unit.all_products_fun import Check
from tests.unit.radar_fun import RadarFun

SCRIPT_PATH = path.dirname(path.realpath(__file__))
FILEPATH = f"{SCRIPT_PATH}/data/galileo/"


class TestGalileo2nc(Check):
    site_meta = {
        "name": "Chilbolton",
        "latitude": 51.144,
        "longitude": -1.439,
        "altitude": 85,
    }
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/galileo.nc"
    uuid = galileo.galileo2nc(FILEPATH, temp_path, site_meta)
    date = "2023-03-08"

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
            "beamwidthH",
            "beamwidthV",
            "antenna_diameter",
            "snr_limit",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_variables(self):
        assert np.isclose(
            self.nc.variables["radar_frequency"][:].data,
            94.0,
            atol=0.1,
        )  # Hard coded
        assert np.all(
            np.isclose(self.nc.variables["zenith_angle"][:].data, 0.0, atol=0.1),
        )

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_long_names(self):
        data = [
            ("SNR", "Signal-to-noise ratio"),
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].long_name
                assert value == expected, f"{value} != {expected}"

    def test_global_attributes(self):
        assert self.nc.source == "RAL Space Galileo"
        assert self.nc.title == f"Galileo cloud radar from {self.site_meta['name']}"

    def test_range(self):
        assert np.min(self.nc.variables["range"][:]) >= 0

    def test_filename_argument(self, tmp_path):
        test_path = tmp_path / "date.nc"
        filename = f"{FILEPATH}galileo-file-1.nc"
        with pytest.raises(RadarDataError):
            galileo.galileo2nc(filename, test_path, self.site_meta)

    def test_correct_date_validation(self, tmp_path):
        test_path = tmp_path / "date.nc"
        galileo.galileo2nc(FILEPATH, test_path, self.site_meta, date=self.date)

    def test_wrong_date_validation(self, tmp_path):
        test_path = tmp_path / "invalid.nc"
        with pytest.raises(ValidTimeStampError):
            galileo.galileo2nc(FILEPATH, test_path, self.site_meta, date="2021-01-03")

    def test_uuid_from_user(self, tmp_path):
        test_path = tmp_path / "uuid.nc"
        uuid_from_user = "fe45561b-eb08-4d2a-a463-c6b4f7be7055"
        uuid = galileo.galileo2nc(
            FILEPATH,
            test_path,
            self.site_meta,
            uuid=uuid_from_user,
        )
        with netCDF4.Dataset(test_path) as nc:
            assert nc.file_uuid == uuid_from_user
            assert str(uuid) == uuid_from_user
