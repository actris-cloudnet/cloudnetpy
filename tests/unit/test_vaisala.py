""" This module contains unit tests for ceilo-module. """
import os
import sys
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest
from all_products_fun import Check
from numpy.testing import assert_equal

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import ceilo2nc, vaisala

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from lidar_fun import LidarFun


@pytest.mark.parametrize(
    "input, result",
    [
        ("01:30:00", 1.5),
        ("02:00:00", 2),
        ("13:15:00", 13.25),
    ],
)
def test_time_to_fraction_hour(input, result):
    assert vaisala.time_to_fraction_hour(input) == result


@pytest.mark.parametrize(
    "keys, values, result",
    [
        (
            ("a", "b"),
            [[1, 2], [1, 2], [1, 2]],
            {"a": np.array([1, 1, 1]), "b": np.array([2, 2, 2])},
        ),
    ],
)
def test_values_to_dict(keys, values, result):
    assert_equal(vaisala.values_to_dict(keys, values), result)


@pytest.mark.parametrize(
    "string, indices, result",
    [
        ("abcd", [3, 4], ["d"]),
        ("abcd", [0, 4], ["abcd"]),
        ("abcdedfg", [1, 2, 4, 5], ["b", "cd", "e"]),
    ],
)
def test_split_string(string, indices, result):
    assert_equal(vaisala.split_string(string, indices), result)


class TestCL51(Check):
    site_meta = {"name": "Kumpula", "altitude": 123, "latitude": 45.0, "longitude": 22.0}
    date = "2020-11-15"
    input = f"{SCRIPT_PATH}/data/vaisala/cl51.DAT"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/cl51.nc"
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
        assert self.nc.variables["wavelength"][:] == 910.0
        assert self.nc.variables["zenith_angle"][:] == 4.5
        assert np.all(np.diff(self.nc.variables["time"][:]) > 0)

    def test_comments(self):
        for key in ("beta", "beta_smooth"):
            assert "SNR threshold applied: 5" in self.nc.variables[key].comment

    def test_global_attributes(self):
        assert self.nc.source == "Vaisala CL51"
        assert self.nc.title == f'CL51 ceilometer from {self.site_meta["name"]}'

    def test_date_argument(self, tmp_path):
        test_path = tmp_path / "date.nc"
        ceilo2nc(self.input, test_path, self.site_meta, date="2020-11-15")
        with netCDF4.Dataset(test_path) as nc:
            assert len(nc.variables["time"]) == 2
            assert nc.year == "2020"
            assert nc.month == "11"
            assert nc.day == "15"
        with pytest.raises(ValidTimeStampError):
            ceilo2nc(self.input, test_path, self.site_meta, date="2021-09-15")


class TestCL31(Check):
    site_meta = {"name": "Kumpula", "altitude": 123, "latitude": 45.0, "longitude": 22.0}
    date = "2020-04-10"
    input = f"{SCRIPT_PATH}/data/vaisala/cl31.DAT"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/cl32.nc"
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
        assert self.nc.variables["wavelength"][:] == 910.0
        assert self.nc.variables["zenith_angle"][:] == 12

    def test_comments(self):
        for key in ("beta", "beta_smooth"):
            assert "SNR threshold applied: 5" in self.nc.variables[key].comment

    def test_global_attributes(self):
        assert self.nc.source == "Vaisala CL31"
        assert self.nc.title == f'CL31 ceilometer from {self.site_meta["name"]}'

    def test_date_argument(self, tmp_path):
        test_path = tmp_path / "date.nc"
        input = f"{SCRIPT_PATH}/data/vaisala/cl31_badtime.DAT"
        ceilo2nc(input, test_path, self.site_meta, date="2020-04-10")
        with netCDF4.Dataset(test_path) as nc:
            assert len(nc.variables["time"]) == 2
            assert nc.year == "2020"
            assert nc.month == "04"
            assert nc.day == "10"
        ceilo2nc(input, test_path, self.site_meta, date="2020-04-11")
        with netCDF4.Dataset(test_path) as nc:
            assert len(nc.variables["time"]) == 2
            assert nc.year == "2020"
            assert nc.month == "04"
            assert nc.day == "11"
        with pytest.raises(ValidTimeStampError):
            ceilo2nc(input, test_path, self.site_meta, date="2020-04-12")


class TestCT25k(Check):
    site_meta = {"name": "Kumpula", "altitude": 123, "latitude": 45.0, "longitude": 22.0}
    date = "2020-10-29"
    input = f"{SCRIPT_PATH}/data/vaisala/ct25k.dat"
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/ct25k.nc"
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
        assert self.nc.variables["wavelength"][:] == 905
        assert self.nc.variables["zenith_angle"][:] == 15

    def test_comments(self):
        for key in ("beta", "beta_smooth"):
            assert "SNR threshold applied: 5" in self.nc.variables[key].comment

    def test_global_attributes(self):
        assert self.nc.source == "Vaisala CT25k"
        assert self.nc.title == f'CT25k ceilometer from {self.site_meta["name"]}'

    def test_date_argument(self, tmp_path):
        test_path = tmp_path / "date.nc"
        ceilo2nc(self.input, test_path, self.site_meta, date="2020-10-29")
        with netCDF4.Dataset(test_path) as nc:
            assert len(nc.variables["time"]) == 3
            assert nc.year == "2020"
            assert nc.month == "10"
            assert nc.day == "29"
        with pytest.raises(ValidTimeStampError):
            ceilo2nc(self.input, test_path, self.site_meta, date="2021-09-15")
