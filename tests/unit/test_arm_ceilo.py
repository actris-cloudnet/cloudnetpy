from os import path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest
from numpy import ma

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import arm_ceilo
from tests.unit.all_products_fun import Check
from tests.unit.lidar_fun import LidarFun

SCRIPT_PATH = path.dirname(path.realpath(__file__))
FILEPATH = f"{SCRIPT_PATH}/data/arm-ceilo/sgpceilC1.b1.20220601.000000.nc"


class TestArmCeilo2nc(Check):
    site_meta = {
        "name": "Southern Great Plains",
        "latitude": 36.605,
        "longitude": -97.485,
        "altitude": 318,
    }
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/ceilo.nc"
    uuid = arm_ceilo.armceilo2nc(FILEPATH, temp_path, site_meta)
    date = "2022-06-01"

    def test_variable_names(self):
        keys = {
            "beta",
            "beta_raw",
            "beta_smooth",
            "calibration_factor",
            "range",
            "height",
            "time",
            "zenith_angle",
            "wavelength",
            "latitude",
            "longitude",
            "altitude",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_common_lidar(self):
        lidar_fun = LidarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in LidarFun.__dict__.items():
            if "test_" in name:
                getattr(lidar_fun, name)()

    def test_variable_values(self):
        assert self.nc.variables["wavelength"][:] == 910
        assert self.nc.variables["zenith_angle"][:] == 1
        assert self.nc.variables["calibration_factor"][:] == 1
        beta = self.nc.variables["beta"][:]
        assert 1e-7 < ma.median(beta) < 1e-4
        assert ma.max(beta) > 1e-4  # cloud

    def test_range(self):
        range_los = self.nc.variables["range"][:]
        assert len(range_los) == 252
        assert range_los[0] == 15
        assert np.all(np.diff(range_los) == 30)

    def test_time(self):
        time = self.nc.variables["time"][:]
        assert len(time) == 120
        assert np.all(np.diff(time) > 0)
        assert 2 <= time[0] < 2.01

    def test_global_attributes(self):
        assert self.nc.source == "Vaisala CL31"
        assert self.nc.title == f"CL31 ceilometer from {self.site_meta['name']}"
        assert self.nc.cloudnet_file_type == "lidar"

    def test_geolocation_from_file(self, tmp_path):
        test_path = tmp_path / "geo.nc"
        arm_ceilo.armceilo2nc(FILEPATH, test_path, {"name": "SGP"})
        with netCDF4.Dataset(test_path) as nc:
            assert np.isclose(nc.variables["latitude"][0], 36.605)
            assert np.isclose(nc.variables["altitude"][0], 318)

    def test_calibration_factor(self, tmp_path):
        test_path = tmp_path / "cal.nc"
        arm_ceilo.armceilo2nc(
            FILEPATH, test_path, {**self.site_meta, "calibration_factor": 2.0}
        )
        with netCDF4.Dataset(test_path) as nc:
            assert nc.variables["calibration_factor"][:] == 2
            assert np.allclose(
                nc.variables["beta_raw"][:], 2 * self.nc.variables["beta_raw"][:]
            )

    def test_wrong_date_validation(self, tmp_path):
        test_path = tmp_path / "invalid.nc"
        with pytest.raises(ValidTimeStampError):
            arm_ceilo.armceilo2nc(
                FILEPATH, test_path, self.site_meta, date="2022-06-02"
            )

    def test_uuid_from_user(self, tmp_path):
        test_path = tmp_path / "uuid.nc"
        uuid_from_user = "fe45561b-eb08-4d2a-a463-c6b4f7be7055"
        uuid = arm_ceilo.armceilo2nc(
            FILEPATH, test_path, self.site_meta, uuid=uuid_from_user
        )
        with netCDF4.Dataset(test_path) as nc:
            assert nc.file_uuid == uuid_from_user
            assert str(uuid) == uuid_from_user
