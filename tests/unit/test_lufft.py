""" This module contains unit tests for ceilo-module. """
from cloudnetpy.instruments import lufft, ceilo2nc
import pytest
import numpy as np
from numpy.testing import assert_array_equal
import netCDF4
import os
import glob
from cloudnetpy import concat_lib
import sys
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy_qc import Quality
from tempfile import NamedTemporaryFile

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from lidar_fun import LidarFun
from all_products_fun import AllProductsFun


@pytest.fixture
def fake_jenoptik_file(tmpdir):
    file_name = tmpdir.join("jenoptik.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    n_time, n_height = 5, 4
    root_grp.createDimension("time", n_time)
    root_grp.createDimension("range", n_height)
    var = root_grp.createVariable("time", "f8", "time")
    var[:] = [3696730788, 3696728448, 3696728447, 3696728450, 3696896790]
    var.units = "seconds since 1904-01-01 00:00:00.000 00:00"
    var = root_grp.createVariable("range", "f8", "range")
    var[:] = np.array([2000, 3000, 4000, 5000])
    var.units = "m"
    var = root_grp.createVariable("beta_raw", "f8", ("time", "range"))
    var[:] = np.random.rand(5, 4)
    var.units = "sr-1 m-1"
    root_grp.createVariable("zenith", "f8")[:] = 2
    root_grp.year = "2021"
    root_grp.month = "2"
    root_grp.day = "21"
    root_grp.software_version = "12.12.1 2.13 1.040 0"
    root_grp.close()
    return file_name


class TestCHM15k:

    date = "2021-02-21"

    @pytest.fixture(autouse=True)
    def init_tests(self, fake_jenoptik_file):
        self.file = fake_jenoptik_file
        self.obj = lufft.LufftCeilo(fake_jenoptik_file, site_meta, self.date)
        self.obj.read_ceilometer_file()

    def test_calc_range(self):
        assert_array_equal(self.obj.data["range"], [1500, 2500, 3500, 4500])

    def test_convert_time(self):
        assert len(self.obj.data["time"]) == 4
        assert all(np.diff(self.obj.data["time"]) > 0)

    def test_read_date(self):
        assert_array_equal(self.obj.date, self.date.split("-"))

    def test_read_metadata(self):
        assert self.obj.data["zenith_angle"] == 2

    def test_convert_time_error(self):
        obj = lufft.LufftCeilo(self.file, site_meta, "2122-01-01")
        with pytest.raises(ValidTimeStampError):
            obj.read_ceilometer_file()


site_meta = {"name": "Bucharest", "altitude": 123, "latitude": 45.0, "longitude": 22.0}
files = glob.glob(f"{SCRIPT_PATH}/data/chm15k/*.nc")
date = "2020-10-22"


class TestWithRealData:

    daily_temp_file = NamedTemporaryFile()
    temp_file = NamedTemporaryFile()
    concat_lib.concatenate_files(files, daily_temp_file.name)
    uuid = ceilo2nc(daily_temp_file.name, temp_file.name, site_meta)

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.nc = netCDF4.Dataset(self.temp_file.name)
        yield
        self.nc.close()

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
        all_fun = AllProductsFun(self.nc, site_meta, date, self.uuid)
        for name, method in AllProductsFun.__dict__.items():
            if "test_" in name:
                getattr(all_fun, name)()

    def test_common_lidar(self):
        lidar_fun = LidarFun(self.nc, site_meta, date, self.uuid)
        for name, method in LidarFun.__dict__.items():
            if "test_" in name:
                getattr(lidar_fun, name)()

    def test_qc(self):
        quality = Quality(self.temp_file.name)
        res_data = quality.check_data()
        res_metadata = quality.check_metadata()
        assert quality.n_metadata_test_failures == 0, res_metadata
        assert quality.n_data_test_failures == 0, res_data

    def test_variable_values(self):
        assert self.nc.variables["wavelength"][:] == 1064
        assert self.nc.variables["zenith_angle"][:] == 0

    def test_comments(self):
        for key in ("beta", "beta_smooth"):
            assert "SNR threshold applied: 5" in self.nc.variables[key].comment

    def test_global_attributes(self):
        assert self.nc.source == "Lufft CHM15k"
        assert self.nc.title == f'CHM15k ceilometer from {site_meta["name"]}'

    def test_date_argument(self):
        output = "asfadfadf"
        ceilo2nc(self.daily_temp_file.name, output, site_meta, date="2020-10-22")
        nc = netCDF4.Dataset(output)
        assert len(nc.variables["time"]) == 20
        assert nc.year == "2020"
        assert nc.month == "10"
        assert nc.day == "22"
        nc.close()
        with pytest.raises(ValidTimeStampError):
            ceilo2nc(self.daily_temp_file.name, self.temp_file.name, site_meta, date="2020-10-23")
        os.remove(output)
