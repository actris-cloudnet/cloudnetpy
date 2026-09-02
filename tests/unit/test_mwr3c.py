from os import path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_allclose

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import mwr3c
from tests.unit.all_products_fun import Check

SCRIPT_PATH = path.dirname(path.realpath(__file__))
FILEPATH = f"{SCRIPT_PATH}/data/mwr3c/enamwr3cC1.b1.20240315.000004.nc"


class TestMwr3c2nc(Check):
    site_meta = {
        "name": "Graciosa",
        "latitude": 39.0916,
        "longitude": -28.0257,
        "altitude": 30,
    }
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/mwr3c.nc"
    uuid = mwr3c.mwr3c2nc(FILEPATH, temp_path, site_meta)
    date = "2024-03-15"

    def test_variable_names(self):
        keys = {
            "time",
            "lwp",
            "iwv",
            "zenith_angle",
            "latitude",
            "longitude",
            "altitude",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_time(self):
        time = self.nc.variables["time"][:]
        assert len(time) == 300
        assert np.all(np.diff(time) > 0)
        assert_allclose(time[:3], np.array([18291, 18293, 18296]) / 3600)

    def test_lwp_and_iwv(self):
        assert_allclose(
            self.nc.variables["lwp"][:3], [0.015892, 0.015678, 0.015678], atol=1e-6
        )
        assert_allclose(
            self.nc.variables["iwv"][:3], [25.82003, 25.902176, 25.902176], atol=1e-4
        )
        assert self.nc.variables["lwp"].units == "kg m-2"
        assert self.nc.variables["iwv"].units == "kg m-2"

    def test_rain_screening(self):
        with netCDF4.Dataset(FILEPATH) as nc:
            is_rain = nc.variables["rain_flag"][:] > 0
        assert is_rain.sum() == 45
        for key in ("lwp", "iwv"):
            mask = ma.getmaskarray(self.nc.variables[key][:])
            assert np.array_equal(mask, is_rain)

    def test_zenith_angle(self):
        assert self.nc.variables["zenith_angle"][:] == 0

    def test_global_attributes(self):
        assert self.nc.source == "Radiometrics MWR-3C"
        assert (
            self.nc.title
            == f"MWR-3C microwave radiometer from {self.site_meta['name']}"
        )
        assert self.nc.cloudnet_file_type == "mwr"
        assert self.nc.serial_number == "R-DPR-07/007"

    def test_qc_and_elevation_screening(self, tmp_path):
        raw_path = tmp_path / "raw.nc"
        with netCDF4.Dataset(FILEPATH) as src, netCDF4.Dataset(raw_path, "w") as dst:
            dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
            dst.createDimension("time", None)
            for name, var in src.variables.items():
                out = dst.createVariable(name, var.dtype, var.dimensions)
                out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                out[:] = var[:]
            dst["qc_tbsky23"][0] = 1
            dst["qc_tbsky31"][1] = 2
            dst["elevation"][2] = 45
            dst["lwp"][3] = -9999
        test_path = tmp_path / "screened.nc"
        mwr3c.mwr3c2nc(raw_path, test_path, self.site_meta)
        with netCDF4.Dataset(test_path) as nc:
            lwp_mask = ma.getmaskarray(nc.variables["lwp"][:])
            iwv_mask = ma.getmaskarray(nc.variables["iwv"][:])
            assert lwp_mask[0] and iwv_mask[0]
            assert lwp_mask[1] and iwv_mask[1]
            assert lwp_mask[2] and iwv_mask[2]
            assert lwp_mask[3] and not iwv_mask[3]
            assert not lwp_mask[4] and not iwv_mask[4]

    @staticmethod
    def _split_file(raw_dir, skip=()):
        parts = []
        raw_dir.mkdir()
        with netCDF4.Dataset(FILEPATH) as src:
            n = len(src.dimensions["time"])
            for i, sl in enumerate((slice(0, n // 2), slice(n // 2, n))):
                part = raw_dir / f"enamwr3cC1.b1.20240315.00000{i}.nc"
                with netCDF4.Dataset(part, "w") as dst:
                    dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
                    dst.createDimension("time", None)
                    for name, var in src.variables.items():
                        if i == 0 and name in skip:
                            continue
                        out = dst.createVariable(name, var.dtype, var.dimensions)
                        out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                        out[:] = var[sl] if "time" in var.dimensions else var[:]
                parts.append(part)
        return parts

    def test_multiple_files(self, tmp_path):
        raw_dir = tmp_path / "raw"
        parts = self._split_file(raw_dir)
        for raw in (parts, raw_dir):
            test_path = tmp_path / "multi.nc"
            mwr3c.mwr3c2nc(raw, test_path, self.site_meta, date=self.date)
            with netCDF4.Dataset(test_path) as nc:
                assert nc.dimensions["time"].size == 300
                assert_allclose(nc.variables["lwp"][:], self.nc.variables["lwp"][:])

    def test_variable_missing_from_first_file(self, tmp_path):
        parts = self._split_file(tmp_path / "raw", skip=("rain_flag",))
        test_path = tmp_path / "multi.nc"
        mwr3c.mwr3c2nc(parts, test_path, self.site_meta, date=self.date)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.dimensions["time"].size == 300

    def test_geolocation_from_file(self, tmp_path):
        test_path = tmp_path / "geo.nc"
        mwr3c.mwr3c2nc(FILEPATH, test_path, {"name": "Graciosa"})
        with netCDF4.Dataset(test_path) as nc:
            assert np.isclose(nc.variables["latitude"][0], 39.0916)
            assert np.isclose(nc.variables["longitude"][0], -28.0257)
            assert np.isclose(nc.variables["altitude"][0], 30.48)

    def test_correct_date_validation(self, tmp_path):
        test_path = tmp_path / "date.nc"
        mwr3c.mwr3c2nc(FILEPATH, test_path, self.site_meta, date=self.date)

    def test_wrong_date_validation(self, tmp_path):
        test_path = tmp_path / "invalid.nc"
        with pytest.raises(ValidTimeStampError):
            mwr3c.mwr3c2nc(FILEPATH, test_path, self.site_meta, date="2024-03-16")

    def test_uuid_from_user(self, tmp_path):
        test_path = tmp_path / "uuid.nc"
        uuid_from_user = "fe45561b-eb08-4d2a-a463-c6b4f7be7055"
        uuid = mwr3c.mwr3c2nc(FILEPATH, test_path, self.site_meta, uuid=uuid_from_user)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.file_uuid == uuid_from_user
            assert str(uuid) == uuid_from_user
