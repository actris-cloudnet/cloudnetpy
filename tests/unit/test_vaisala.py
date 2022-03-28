""" This module contains unit tests for ceilo-module. """
import os
from cloudnetpy.instruments import vaisala, ceilo2nc
import pytest
import numpy as np
from numpy.testing import assert_equal
import netCDF4
import sys
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy_qc import Quality

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from lidar_fun import LidarFun
from all_products_fun import AllProductsFun


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


site_meta = {"name": "Kumpula", "altitude": 123, "latitude": 45.0, "longitude": 22.0}


class TestCL51:
    date = "2020-11-15"
    input = f"{SCRIPT_PATH}/data/vaisala/cl51.DAT"
    output = "dummy_cl51_file.nc"
    uuid = ceilo2nc(input, output, site_meta)
    nc = netCDF4.Dataset(output)
    lidar_fun = LidarFun(nc, site_meta, date, uuid)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)
    quality = Quality(output)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()

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

    def test_common(self):
        for name, method in AllProductsFun.__dict__.items():
            if "test_" in name:
                getattr(self.all_fun, name)()

    def test_common_lidar(self):
        for name, method in LidarFun.__dict__.items():
            if "test_" in name:
                getattr(self.lidar_fun, name)()

    def test_qc(self):
        assert self.quality.n_metadata_test_failures == 0, self.res_metadata
        assert self.quality.n_data_test_failures == 0, self.res_data

    def test_variable_values(self):
        assert self.nc.variables["wavelength"][:] == 910.0
        assert self.nc.variables["zenith_angle"][:] == 4.5
        assert np.all(np.diff(self.nc.variables["time"][:]) > 0)

    def test_comments(self):
        for key in ("beta", "beta_smooth"):
            assert "SNR threshold applied: 5" in self.nc.variables[key].comment

    def test_global_attributes(self):
        assert self.nc.source == "Vaisala CL51"
        assert self.nc.title == f'CL51 ceilometer from {site_meta["name"]}'

    def test_date_argument(self):
        output = "dummy_sdfsdf_output_file.nc"
        ceilo2nc(self.input, output, site_meta, date="2020-11-15")
        nc = netCDF4.Dataset(output)
        assert len(nc.variables["time"]) == 2
        assert nc.year == "2020"
        assert nc.month == "11"
        assert nc.day == "15"
        nc.close()
        with pytest.raises(ValidTimeStampError):
            ceilo2nc(self.input, output, site_meta, date="2021-09-15")
        os.remove(output)

    def test_cleanup(self):
        os.remove(self.output)
        self.nc.close()


class TestCL31:
    date = "2020-04-10"
    input = f"{SCRIPT_PATH}/data/vaisala/cl31.DAT"
    output = "dummy_cl31_file.nc"
    uuid = ceilo2nc(input, output, site_meta)
    nc = netCDF4.Dataset(output)
    lidar_fun = LidarFun(nc, site_meta, date, uuid)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)
    quality = Quality(output)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()

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

    def test_common(self):
        for name, method in AllProductsFun.__dict__.items():
            if "test_" in name:
                getattr(self.all_fun, name)()

    def test_common_lidar(self):
        for name, method in LidarFun.__dict__.items():
            if "test_" in name:
                getattr(self.lidar_fun, name)()

    def test_qc(self):
        assert self.quality.n_metadata_test_failures == 0, self.res_metadata
        assert self.quality.n_data_test_failures == 0, self.res_data

    def test_variable_values(self):
        assert self.nc.variables["wavelength"][:] == 910.0
        assert self.nc.variables["zenith_angle"][:] == 12

    def test_comments(self):
        for key in ("beta", "beta_smooth"):
            assert "SNR threshold applied: 5" in self.nc.variables[key].comment

    def test_global_attributes(self):
        assert self.nc.source == "Vaisala CL31"
        assert self.nc.title == f'CL31 ceilometer from {site_meta["name"]}'

    def test_date_argument(self):
        output = "falskdfjlskdf"
        input = f"{SCRIPT_PATH}/data/vaisala/cl31_badtime.DAT"
        ceilo2nc(input, output, site_meta, date="2020-04-10")
        nc = netCDF4.Dataset(output)
        assert len(nc.variables["time"]) == 2
        assert nc.year == "2020"
        assert nc.month == "04"
        assert nc.day == "10"
        nc.close()
        ceilo2nc(input, output, site_meta, date="2020-04-11")
        nc = netCDF4.Dataset(output)
        assert len(nc.variables["time"]) == 2
        assert nc.year == "2020"
        assert nc.month == "04"
        assert nc.day == "11"
        nc.close()
        with pytest.raises(ValidTimeStampError):
            ceilo2nc(input, output, site_meta, date="2020-04-12")
        os.remove(output)

    def test_cleanup(self):
        os.remove(self.output)
        self.nc.close()


class TestCT25k:
    date = "2020-10-29"
    input = f"{SCRIPT_PATH}/data/vaisala/ct25k.dat"
    output = "dummy_ct25k_file.nc"
    uuid = ceilo2nc(input, output, site_meta)
    nc = netCDF4.Dataset(output)
    lidar_fun = LidarFun(nc, site_meta, date, uuid)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)
    quality = Quality(output)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()

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

    def test_common(self):
        for name, method in AllProductsFun.__dict__.items():
            if "test_" in name:
                getattr(self.all_fun, name)()

    def test_common_lidar(self):
        for name, method in LidarFun.__dict__.items():
            if "test_" in name:
                getattr(self.lidar_fun, name)()

    def test_qc(self):
        assert self.quality.n_metadata_test_failures == 0, self.res_metadata
        assert self.quality.n_data_test_failures == 0, self.res_data

    def test_variable_values(self):
        assert self.nc.variables["wavelength"][:] == 905
        assert self.nc.variables["zenith_angle"][:] == 15

    def test_comments(self):
        for key in ("beta", "beta_smooth"):
            assert "SNR threshold applied: 5" in self.nc.variables[key].comment

    def test_global_attributes(self):
        assert self.nc.source == "Vaisala CT25k"
        assert self.nc.title == f'CT25k ceilometer from {site_meta["name"]}'

    def test_date_argument(self):
        output = "falskdfjlskdf"
        ceilo2nc(self.input, output, site_meta, date="2020-10-29")
        nc = netCDF4.Dataset(output)
        assert len(nc.variables["time"]) == 3
        assert nc.year == "2020"
        assert nc.month == "10"
        assert nc.day == "29"
        nc.close()
        with pytest.raises(ValidTimeStampError):
            ceilo2nc(self.input, output, site_meta, date="2021-09-15")
        os.remove(output)

    def test_cleanup(self):
        os.remove(self.output)
        self.nc.close()
