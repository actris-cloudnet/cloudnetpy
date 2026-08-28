from os import path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest
from numpy.testing import assert_allclose

from cloudnetpy.exceptions import DisdrometerDataError
from cloudnetpy.instruments import arm_ld
from tests.unit.all_products_fun import Check

SCRIPT_PATH = path.dirname(path.realpath(__file__))
FILEPATH = f"{SCRIPT_PATH}/data/arm-ld/sgpldC1.b1.20220601.000000.cdf"


class TestArmLd2nc(Check):
    site_meta = {
        "name": "Southern Great Plains",
        "latitude": 36.605,
        "longitude": -97.485,
        "altitude": 318,
    }
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/ld.nc"
    uuid = arm_ld.armld2nc(FILEPATH, temp_path, site_meta)
    date = "2022-06-01"

    def test_variable_names(self):
        keys = {
            "time",
            "interval",
            "data_raw",
            "n_particles",
            "number_concentration",
            "fall_velocity",
            "rainfall_rate",
            "rainfall_amount",
            "radar_reflectivity",
            "kinetic_energy",
            "visibility",
            "synop_WaWa",
            "diameter",
            "diameter_spread",
            "diameter_bnds",
            "velocity",
            "velocity_spread",
            "velocity_bnds",
            "nominal_area",
            "effective_area",
            "latitude",
            "longitude",
            "altitude",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_dimensions(self):
        assert self.nc.dimensions["time"].size == 90
        assert self.nc.dimensions["diameter"].size == 32
        assert self.nc.dimensions["velocity"].size == 32
        assert self.nc.variables["data_raw"].shape == (90, 32, 32)

    def test_rainfall_rate_matches_arm(self):
        with netCDF4.Dataset(FILEPATH) as raw:
            arm_rate = raw.variables["precip_rate"][:]  # mm/h
            arm_wawa = raw.variables["weather_code"][:]
        rate = self.nc.variables["rainfall_rate"][:] * 3600 * 1000
        assert_allclose(rate, arm_rate, rtol=0.05, atol=0.02)
        assert rate.max() > 5
        assert np.array_equal(self.nc.variables["synop_WaWa"][:], arm_wawa)

    def test_time(self):
        time = self.nc.variables["time"][:]
        assert np.all(np.diff(time) > 0)
        assert np.isclose(time[0], 16.7833, atol=1e-3)
        assert np.all(self.nc.variables["interval"][:] == 60)

    def test_global_attributes(self):
        assert self.nc.source == "OTT HydroMet Parsivel2"
        assert self.nc.title == f"Parsivel2 disdrometer from {self.site_meta['name']}"
        assert self.nc.cloudnet_file_type == "disdrometer"

    def test_geolocation_from_file(self, tmp_path):
        test_path = tmp_path / "geo.nc"
        arm_ld.armld2nc(FILEPATH, test_path, {"name": "SGP"})
        with netCDF4.Dataset(test_path) as nc:
            assert np.allclose(nc.variables["latitude"][:], 36.605, atol=0.01)
            assert np.allclose(nc.variables["altitude"][:], 318, atol=1)

    def test_wrong_date_validation(self, tmp_path):
        test_path = tmp_path / "invalid.nc"
        with pytest.raises(DisdrometerDataError):
            arm_ld.armld2nc(FILEPATH, test_path, self.site_meta, date="2022-06-02")

    def test_uuid_from_user(self, tmp_path):
        test_path = tmp_path / "uuid.nc"
        uuid_from_user = "fe45561b-eb08-4d2a-a463-c6b4f7be7055"
        uuid = arm_ld.armld2nc(FILEPATH, test_path, self.site_meta, uuid=uuid_from_user)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.file_uuid == uuid_from_user
            assert str(uuid) == uuid_from_user
