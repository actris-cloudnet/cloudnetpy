from pathlib import Path

import netCDF4
import numpy as np
import pytest
from cloudnetpy_qc import Quality

SITE_META = {"name": "Kumpula", "altitude": 50, "latitude": 23, "longitude": 34.0}


class Check:

    temp_path: str
    nc: netCDF4.Dataset
    date: str
    site_meta: dict
    uuid: str

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        self.nc = netCDF4.Dataset(self.temp_path)
        yield
        self.nc.close()

    def test_qc(self):
        quality = Quality(self.temp_path)
        res_data = quality.check_data()
        res_metadata = quality.check_metadata()
        assert quality.n_metadata_test_failures == 0, res_metadata
        assert quality.n_data_test_failures == 0, res_data

    def test_common(self):
        all_fun = AllProductsFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in AllProductsFun.__dict__.items():
            if "test_" in name:
                getattr(all_fun, name)()


class AllProductsFun:
    """Common tests for all Cloudnet products."""

    def __init__(self, nc: netCDF4.Dataset, site_meta: dict, date: str, uuid):
        self.nc = nc
        self.site_meta = site_meta
        self.date = date
        self.uuid = uuid

    def test_variable_names(self):
        keys = {"time", "latitude", "longitude", "altitude"}
        for key in keys:
            assert key in self.nc.variables

    def test_nan_values(self):
        for key in self.nc.variables.keys():
            assert bool(np.isnan(self.nc.variables[key]).all()) is False

    def test_time_axis(self):
        assert self.nc.variables["time"].axis == "T"

    def test_empty_units(self):
        for key in self.nc.variables.keys():
            if hasattr(self.nc.variables[key], "units"):
                value = self.nc.variables[key].units
                assert value != "", f"{key} - {value}"

    def test_variable_values(self):
        for key in ("altitude", "latitude", "longitude"):
            value = self.nc.variables[key][:]
            expected = self.site_meta[key]
            assert value == expected, f"{value} != {expected}"

    def test_invalid_units(self):
        for key in self.nc.variables:
            variable = self.nc.variables[key]
            assert hasattr(variable, "units")
            assert variable.units != "", key

    def test_units(self):
        """Custom units that are not tested in QC tests."""
        data = [
            ("time", f"hours since {self.date} 00:00:00 +00:00"),
        ]
        for key, expected in data:
            if key in self.nc.variables:
                value = self.nc.variables[key].units
                assert value == expected, f"{value} != {expected}"

    def test_long_name_format(self):
        for key in self.nc.variables:
            assert hasattr(self.nc.variables[key], "long_name")
            value = self.nc.variables[key].long_name
            assert not value.endswith(".")

    def test_global_attributes(self):
        assert self.nc.location == self.site_meta["name"]
        assert self.nc.file_uuid == self.uuid
        assert self.nc.Conventions == "CF-1.8"
        y, m, d = self.date.split("-")
        assert self.nc.year == y
        assert self.nc.month == m
        assert self.nc.day == d
        for key in ("cloudnetpy_version", "references", "history"):
            assert hasattr(self.nc, key)
