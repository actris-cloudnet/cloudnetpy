import os
import sys
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import numpy.ma as ma
import pytest

import cloudnetpy.instruments.pollyxt
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import pollyxt2nc
from tests.unit.all_products_fun import SITE_META, Check
from tests.unit.lidar_fun import LidarFun

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
filepath = f"{SCRIPT_PATH}/data/pollyxt/"


class TestPolly(Check):
    date = "2021-09-17"
    site_meta = SITE_META
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/polly.nc"
    uuid = pollyxt2nc(filepath, temp_path, site_meta)

    def test_variable_names(self):
        keys = {
            "beta",
            "beta_raw",
            "calibration_factor",
            "range",
            "height",
            "zenith_angle",
            "time",
            "depolarisation",
            "depolarisation_raw",
            "altitude",
            "latitude",
            "longitude",
            "wavelength",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_common_lidar(self):
        lidar_fun = LidarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in LidarFun.__dict__.items():
            if "test_" in name:
                getattr(lidar_fun, name)()

    def test_variable_values(self):
        assert self.nc.variables["wavelength"][:] == 1064.0
        assert self.nc.variables["zenith_angle"][:] == 5.0
        assert ma.max(self.nc.variables["depolarisation"][:]) < 1
        assert ma.min(self.nc.variables["depolarisation"][:]) > -0.1
        assert np.all(np.diff(self.nc.variables["time"][:]) > 0)

    def test_comments(self):
        assert "SNR threshold applied: 2" in self.nc.variables["beta"].comment

    def test_global_attributes(self):
        assert self.nc.source == "TROPOS PollyXT"
        assert self.nc.title == f"PollyXT Raman lidar from {self.site_meta['name']}"


class TestPolly2:
    def test_date_argument(self, tmp_path):
        temp_path = tmp_path / "date.nc"
        pollyxt2nc(filepath, temp_path, SITE_META, date="2021-09-17")
        with netCDF4.Dataset(temp_path) as nc:
            assert len(nc.variables["time"]) == 80
            assert nc.year == "2021"
            assert nc.month == "09"
            assert nc.day == "17"
        with pytest.raises(ValidTimeStampError):
            pollyxt2nc(filepath, temp_path, SITE_META, date="2021-09-15")

    def test_snr_limit(self, tmp_path):
        temp_path = tmp_path / "limit.nc"
        meta = SITE_META.copy()
        meta["snr_limit"] = 3.2
        pollyxt2nc(filepath, temp_path, meta, date="2021-09-17")
        with netCDF4.Dataset(temp_path) as nc:
            assert "SNR threshold applied: 3.2" in nc.variables["beta"].comment

    def test_site_meta(self, tmp_path):
        temp_path = tmp_path / "meta.nc"
        meta = {"name": "Mindelo", "altitude": 123, "kissa": 34}
        pollyxt2nc(filepath, temp_path, meta, date="2021-09-17")
        with netCDF4.Dataset(temp_path) as nc:
            assert "altitude" in nc.variables
            for key in ("latitude", "longitude", "kissa"):
                assert key not in nc.variables


def test_broken_channel():
    filepath = f"{SCRIPT_PATH}/data/pollyxt/broken_1064_channel"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/polly.nc"
    pollyxt2nc(filepath, temp_path, SITE_META, date="2022-06-16")
    with netCDF4.Dataset(temp_path) as nc:
        assert nc.variables["wavelength"][:] == 532


@pytest.mark.parametrize(
    "data, output",
    [
        (ma.array([1, 2, 3], mask=[1, 1, 1]), True),
        (ma.array([1, 2, 3], mask=[1, 1, 0]), False),
        (ma.array([0, 0, 0], mask=[0, 0, 0]), True),
        (ma.array([0, 2, 0], mask=[0, 0, 0]), False),
    ],
)
def test_is_zeros_or_masked(data, output):
    assert cloudnetpy.instruments.pollyxt._only_zeros_or_masked(data) == output
