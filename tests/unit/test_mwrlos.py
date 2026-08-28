from os import path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_allclose

from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import mwrlos
from tests.unit.all_products_fun import Check

SCRIPT_PATH = path.dirname(path.realpath(__file__))
FILEPATH = f"{SCRIPT_PATH}/data/mwrlos/sgpmwrlosC1.b1.20100310.000025.cdf"


class TestMwrlos2nc(Check):
    site_meta = {
        "name": "Southern Great Plains",
        "latitude": 36.606,
        "longitude": -97.485,
        "altitude": 316,
    }
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/mwrlos.nc"
    uuid = mwrlos.mwrlos2nc(FILEPATH, temp_path, site_meta)
    date = "2010-03-10"

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
        assert_allclose(time[:3], np.array([25, 44, 74]) / 3600)

    def test_lwp_and_iwv(self):
        assert_allclose(self.nc.variables["lwp"][:3], [0.017, 0.019, 0.02], atol=1e-6)
        assert_allclose(self.nc.variables["iwv"][:3], [13.29, 13.2, 13.32], atol=1e-4)
        assert self.nc.variables["lwp"].units == "kg m-2"
        assert self.nc.variables["iwv"].units == "kg m-2"

    def test_zenith_angle(self):
        assert self.nc.variables["zenith_angle"][:] == 0

    def test_global_attributes(self):
        assert self.nc.source == "Radiometrics WVR-1100"
        assert (
            self.nc.title
            == f"WVR-1100 microwave radiometer from {self.site_meta['name']}"
        )
        assert self.nc.cloudnet_file_type == "mwr"
        assert self.nc.serial_number == "10"

    def test_qc_and_wet_window_screening(self, tmp_path):
        raw_path = tmp_path / "raw.cdf"
        with netCDF4.Dataset(FILEPATH) as src, netCDF4.Dataset(raw_path, "w") as dst:
            dst.setncatts({k: src.getncattr(k) for k in src.ncattrs()})
            dst.createDimension("time", None)
            for name, var in src.variables.items():
                out = dst.createVariable(name, var.dtype, var.dimensions)
                out.setncatts({k: var.getncattr(k) for k in var.ncattrs()})
                out[:] = var[:]
            dst["qc_liq"][0] = 1
            dst["qc_vap"][1] = 2
            dst["wet_window"][2] = 1
        test_path = tmp_path / "screened.nc"
        mwrlos.mwrlos2nc(raw_path, test_path, self.site_meta)
        with netCDF4.Dataset(test_path) as nc:
            lwp_mask = ma.getmaskarray(nc.variables["lwp"][:])
            iwv_mask = ma.getmaskarray(nc.variables["iwv"][:])
            assert lwp_mask[0] and not iwv_mask[0]
            assert iwv_mask[1] and not lwp_mask[1]
            assert lwp_mask[2] and iwv_mask[2]
            assert not lwp_mask[3] and not iwv_mask[3]

    @staticmethod
    def _split_file(raw_dir, skip=()):
        parts = []
        raw_dir.mkdir()
        with netCDF4.Dataset(FILEPATH) as src:
            n = len(src.dimensions["time"])
            for i, sl in enumerate((slice(0, n // 2), slice(n // 2, n))):
                part = raw_dir / f"sgpmwrlosC1.b1.20100310.00000{i}.cdf"
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
            mwrlos.mwrlos2nc(raw, test_path, self.site_meta, date=self.date)
            with netCDF4.Dataset(test_path) as nc:
                assert nc.dimensions["time"].size == 300
                assert_allclose(nc.variables["lwp"][:], self.nc.variables["lwp"][:])

    def test_variable_missing_from_first_file(self, tmp_path):
        parts = self._split_file(tmp_path / "raw", skip=("wet_window",))
        test_path = tmp_path / "multi.nc"
        mwrlos.mwrlos2nc(parts, test_path, self.site_meta, date=self.date)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.dimensions["time"].size == 300

    def test_geolocation_from_file(self, tmp_path):
        test_path = tmp_path / "geo.nc"
        mwrlos.mwrlos2nc(FILEPATH, test_path, {"name": "SGP"})
        with netCDF4.Dataset(test_path) as nc:
            assert np.isclose(nc.variables["latitude"][0], 36.606)
            assert np.isclose(nc.variables["longitude"][0], -97.485)
            assert np.isclose(nc.variables["altitude"][0], 316)

    def test_correct_date_validation(self, tmp_path):
        test_path = tmp_path / "date.nc"
        mwrlos.mwrlos2nc(FILEPATH, test_path, self.site_meta, date=self.date)

    def test_wrong_date_validation(self, tmp_path):
        test_path = tmp_path / "invalid.nc"
        with pytest.raises(ValidTimeStampError):
            mwrlos.mwrlos2nc(FILEPATH, test_path, self.site_meta, date="2010-03-11")

    def test_uuid_from_user(self, tmp_path):
        test_path = tmp_path / "uuid.nc"
        uuid_from_user = "fe45561b-eb08-4d2a-a463-c6b4f7be7055"
        uuid = mwrlos.mwrlos2nc(
            FILEPATH, test_path, self.site_meta, uuid=uuid_from_user
        )
        with netCDF4.Dataset(test_path) as nc:
            assert nc.file_uuid == uuid_from_user
            assert str(uuid) == uuid_from_user
