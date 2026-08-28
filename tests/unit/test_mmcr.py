from os import path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest
from numpy import ma

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import mmcr
from tests.unit.all_products_fun import Check
from tests.unit.radar_fun import RadarFun

SCRIPT_PATH = path.dirname(path.realpath(__file__))
FILEPATH = f"{SCRIPT_PATH}/data/mmcr/sgpmmcrmomC1.b1.20100310.000047.cdf"


class TestMmcr2nc(Check):
    site_meta = {
        "name": "Southern Great Plains",
        "latitude": 36.606,
        "longitude": -97.485,
        "altitude": 316,
    }
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/mmcr.nc"
    uuid = mmcr.mmcr2nc(FILEPATH, temp_path, site_meta)
    date = "2010-03-10"

    def test_variable_names(self):
        keys = {
            "Zh",
            "v",
            "width",
            "SNR",
            "correction_bits",
            "time",
            "range",
            "radar_frequency",
            "nyquist_velocity",
            "snr_limit",
            "latitude",
            "longitude",
            "altitude",
            "zenith_angle",
            "height",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_variables(self):
        assert np.isclose(self.nc.variables["radar_frequency"][:].data, 34.86)
        assert np.isclose(
            self.nc.variables["nyquist_velocity"][:].data, 5.02, atol=0.01
        )
        assert self.nc.variables["zenith_angle"][:] == 0
        assert -20 < self.nc.variables["snr_limit"][:] < -10

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_global_attributes(self):
        assert self.nc.source == "MMCR"
        assert self.nc.title == f"MMCR cloud radar from {self.site_meta['name']}"

    def test_range(self):
        assert self.nc.dimensions["range"].size == 167
        assert np.isclose(self.nc.variables["range"][0], 75.68, atol=0.01)
        assert np.all(np.diff(self.nc.variables["range"][:]) > 0)
        assert np.allclose(
            self.nc.variables["height"][:],
            self.nc.variables["range"][:] + self.site_meta["altitude"],
        )

    def test_only_selected_mode(self):
        assert self.nc.dimensions["time"].size == 141
        assert np.all(np.diff(self.nc.variables["time"][:]) > 0)

    def test_velocity_sign(self):
        # Falling ice cloud should have negative velocity
        v = self.nc.variables["v"][:]
        height = self.nc.variables["range"][:]
        ice = v[:, (height > 3000) & (height < 6000)]
        assert ma.median(ice) < -0.3

    def test_dealiased(self):
        assert "Dealiased" in self.nc.variables["v"].comment
        v = self.nc.variables["v"][:]
        nyquist = self.nc.variables["nyquist_velocity"][:]
        assert ma.min(v) >= -3 * nyquist
        assert ma.max(v) <= nyquist
        assert ma.median(v) < 0

    def test_snr_screening(self):
        snr = self.nc.variables["SNR"][:]
        assert ma.min(snr) >= self.nc.variables["snr_limit"][:] - 3
        for key in ("Zh", "v", "width"):
            mask = ma.getmaskarray(self.nc.variables[key][:])
            assert np.all(mask[ma.getmaskarray(snr)])

    def test_other_mode(self, tmp_path):
        test_path = tmp_path / "ci.nc"
        mmcr.mmcr2nc(FILEPATH, test_path, {**self.site_meta, "mode": "CI"})
        with netCDF4.Dataset(test_path) as nc:
            assert nc.dimensions["time"].size == 71
            assert np.isclose(nc.variables["nyquist_velocity"][:].data, 4.27, atol=0.01)

    def test_invalid_mode(self, tmp_path):
        test_path = tmp_path / "invalid.nc"
        with pytest.raises(ValueError, match="not found"):
            mmcr.mmcr2nc(FILEPATH, test_path, {**self.site_meta, "mode": "XX"})

    def test_geolocation_from_file(self, tmp_path):
        test_path = tmp_path / "geo.nc"
        mmcr.mmcr2nc(FILEPATH, test_path, {"name": "SGP"})
        with netCDF4.Dataset(test_path) as nc:
            assert np.isclose(nc.variables["latitude"][0], 36.606)
            assert np.isclose(nc.variables["longitude"][0], -97.485)
            assert np.isclose(nc.variables["altitude"][0], 316)

    def test_correct_date_validation(self, tmp_path):
        test_path = tmp_path / "date.nc"
        mmcr.mmcr2nc(FILEPATH, test_path, self.site_meta, date=self.date)

    def test_wrong_date_validation(self, tmp_path):
        test_path = tmp_path / "invalid.nc"
        with pytest.raises(ValidTimeStampError):
            mmcr.mmcr2nc(FILEPATH, test_path, self.site_meta, date="2010-03-11")

    def test_uuid_from_user(self, tmp_path):
        test_path = tmp_path / "uuid.nc"
        uuid_from_user = "fe45561b-eb08-4d2a-a463-c6b4f7be7055"
        uuid = mmcr.mmcr2nc(FILEPATH, test_path, self.site_meta, uuid=uuid_from_user)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.file_uuid == uuid_from_user
            assert str(uuid) == uuid_from_user
