import glob
import os
import sys
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import numpy.ma as ma

from cloudnetpy import concat_lib
from cloudnetpy.instruments import ceilo2nc
from tests.unit.all_products_fun import Check
from tests.unit.lidar_fun import LidarFun

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
FILES = glob.glob(f"{SCRIPT_PATH}/data/cl61d/*.nc")
FILES.sort()
FILES2 = glob.glob(f"{SCRIPT_PATH}/data/cl61d-2/*.nc")
FILES2.sort()

SITE_META = {
    "name": "Hyytiälä",
    "altitude": 123,
    "calibration_factor": 2.0,
    "latitude": 45.0,
    "longitude": 22.0,
}

VARIABLES = {
    "beta",
    "beta_smooth",
    "calibration_factor",
    "range",
    "height",
    "zenith_angle",
    "time",
    "depolarisation",
    "altitude",
    "latitude",
    "longitude",
    "wavelength",
}


class TestCl61d(Check):
    site_meta = SITE_META
    date = "2021-08-29"
    temp_dir = TemporaryDirectory()
    daily_file = temp_dir.name + "/daily.nc"
    concat_lib.concatenate_files(FILES, daily_file, concat_dimension="profile")
    temp_path = temp_dir.name + "/test.nc"
    uuid = ceilo2nc(daily_file, temp_path, site_meta, date=date)

    def test_variable_names(self):
        assert set(self.nc.variables.keys()) == VARIABLES

    def test_common_lidar(self):
        lidar_fun = LidarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in LidarFun.__dict__.items():
            if "test_" in name:
                getattr(lidar_fun, name)()

    def test_variable_values(self):
        assert abs(self.nc.variables["wavelength"][:] - 910.55) < 0.001
        assert self.nc.variables["zenith_angle"][:] == 3.0
        assert ma.max(self.nc.variables["depolarisation"][:]) < 1
        assert ma.min(self.nc.variables["depolarisation"][:]) > -0.1

    def test_comments(self):
        assert "SNR threshold applied: 5" in self.nc.variables["beta"].comment


class TestCl61d2(Check):
    site_meta = {**SITE_META, "model": "cl61d"}
    date = "2023-07-30"
    temp_dir = TemporaryDirectory()
    daily_file = temp_dir.name + "/daily.nc"
    concat_lib.concatenate_files(FILES2, daily_file, concat_dimension="time")
    temp_path = temp_dir.name + "/test.nc"
    uuid = ceilo2nc(daily_file, temp_path, site_meta, date=date)

    def test_variable_names(self):
        assert set(self.nc.variables.keys()) == VARIABLES

    def test_common_lidar(self):
        lidar_fun = LidarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in LidarFun.__dict__.items():
            if "test_" in name:
                getattr(lidar_fun, name)()

    def test_variable_values(self):
        assert abs(self.nc.variables["wavelength"][:] - 910.55) < 0.001
        assert abs(self.nc.variables["zenith_angle"][:] - 3.4) < 0.001
        assert ma.max(self.nc.variables["depolarisation"][:]) < 1
        assert ma.min(self.nc.variables["depolarisation"][:]) > -0.1

    def test_comments(self):
        assert "SNR threshold applied: 5" in self.nc.variables["beta"].comment


def test_date_argument(tmp_path):
    daily_file = str(tmp_path / "daily.nc")
    test_file = str(tmp_path / "test.nc")
    concat_lib.concatenate_files(FILES, daily_file, concat_dimension="profile")
    ceilo2nc(daily_file, test_file, SITE_META, date="2021-08-30")
    with netCDF4.Dataset(test_file) as nc:
        assert len(nc.variables["time"]) == 12
        assert np.all(np.diff(nc.variables["time"][:]) > 0)
        assert nc.year == "2021"
        assert nc.month == "08"
        assert nc.day == "30"
