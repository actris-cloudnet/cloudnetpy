import netCDF4
import numpy as np
import numpy.ma as ma


class LidarFun:
    """Common tests for all lidars."""

    def __init__(self, nc: netCDF4.Dataset, site_meta: dict, date: str, uuid):
        self.nc = nc
        self.site_meta = site_meta
        self.date = date
        self.uuid = uuid

    def test_data_types(self):
        for key in self.nc.variables.keys():
            value = self.nc.variables[key].dtype
            assert value == "float32", f"{value} - {key}"

    def test_axis(self):
        assert self.nc.variables["range"].axis == "Z"
        for key in self.nc.variables.keys():
            if key not in ("time", "range"):
                assert hasattr(self.nc.variables[key], "axis") is False

    def test_variable_values(self):
        assert 900 < self.nc.variables["wavelength"][:] < 1065
        assert 0 <= self.nc.variables["zenith_angle"][:] < 90
        assert np.all(
            (
                self.nc.variables["height"][:]
                - self.site_meta["altitude"]
                - self.nc.variables["range"][:]
            )
            <= 1e-3
        )
        assert ma.min(self.nc.variables["beta"][:]) > 0

    def test_global_attributes(self):
        assert self.nc.cloudnet_file_type == "lidar"
