import os
import glob
import pytest
from cloudnetpy import concat_lib
from cloudnetpy.instruments import ceilo2nc
import netCDF4
import numpy as np
import numpy.ma as ma
import sys
from cloudnetpy_qc import Quality
from tempfile import NamedTemporaryFile

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from lidar_fun import LidarFun
from all_products_fun import AllProductsFun

site_meta = {
    "name": "Hyytiälä",
    "altitude": 123,
    "calibration_factor": 2.0,
    "latitude": 45.0,
    "longitude": 22.0,
}
files = glob.glob(f"{SCRIPT_PATH}/data/cl61d/*.nc")
files.sort()
daily_file = "dummy_cl61_daily_file.nc"
concat_lib.concatenate_files(files, daily_file, concat_dimension="profile")
date = "2021-08-29"


class TestCl61d:

    temp_file = NamedTemporaryFile()
    uuid = ceilo2nc(daily_file, temp_file.name, site_meta, date=date)
    quality = Quality(temp_file.name)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.nc = netCDF4.Dataset(self.temp_file.name)
        yield
        self.nc.close()

    def test_variable_names(self):
        keys = {
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
        assert self.quality.n_metadata_test_failures == 0, self.res_metadata
        assert self.quality.n_data_test_failures == 0, self.res_data

    def test_variable_values(self):
        assert abs(self.nc.variables["wavelength"][:] - 910.55) < 0.001
        assert self.nc.variables["zenith_angle"][:] == 3.0
        assert ma.max(self.nc.variables["depolarisation"][:]) < 1
        assert ma.min(self.nc.variables["depolarisation"][:]) > -0.1

    def test_comments(self):
        assert "SNR threshold applied: 5" in self.nc.variables["beta"].comment

    def test_global_attributes(self):
        assert self.nc.source == "Vaisala CL61d"
        assert self.nc.title == f'CL61d ceilometer from {site_meta["name"]}'


def test_date_argument():
    output = "dummy_asdfasdfa_output_file.nc"
    concat_lib.concatenate_files(files, daily_file, concat_dimension="profile")
    ceilo2nc(daily_file, output, site_meta, date="2021-08-30")
    nc = netCDF4.Dataset(output)
    assert len(nc.variables["time"]) == 12
    assert np.all(np.diff(nc.variables["time"][:]) > 0)
    assert nc.year == "2021"
    assert nc.month == "08"
    assert nc.day == "30"
    nc.close()
    os.remove(output)
    os.remove(daily_file)
