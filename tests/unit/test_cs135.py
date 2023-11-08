""" This module contains unit tests for ceilo-module. """
import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import ceilo2nc
from tests.unit.all_products_fun import Check
from tests.unit.lidar_fun import LidarFun

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class TestCS135(Check):
    site_meta = {
        "name": "Kumpula",
        "altitude": 123,
        "latitude": 45.0,
        "longitude": 22.0,
        "model": "cs135",
    }
    date = "2023-06-12"
    input = f"{SCRIPT_PATH}/data/cs135/20230612_ceilometer.txt"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/cs135.nc"
    uuid = ceilo2nc(input, temp_path, site_meta)

    def test_variable_names(self):
        keys = {
            "beta",
            "beta_raw",
            "beta_smooth",
            "calibration_factor",
            "range",
            "height",
            "zenith_angle",
            "time",
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
        assert self.nc.variables["wavelength"][:] == 905.0
        assert self.nc.variables["zenith_angle"][:] == 2.0
        assert np.all(np.diff(self.nc.variables["time"][:]) > 0)

    def test_comments(self):
        for key in ("beta", "beta_smooth"):
            assert "SNR threshold applied: 5" in self.nc.variables[key].comment

    def test_global_attributes(self):
        assert self.nc.source == "Campbell Scientific CS135"
        assert self.nc.title == f'CS135 ceilometer from {self.site_meta["name"]}'

    def test_date_argument(self, tmp_path):
        with pytest.raises(ValidTimeStampError):
            ceilo2nc(self.input, "/tmp/foo.nc", self.site_meta, date="2021-09-15")

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 8
