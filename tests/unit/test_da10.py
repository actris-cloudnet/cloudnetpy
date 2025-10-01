import glob
import os
from tempfile import TemporaryDirectory

from cloudnetpy import concat_lib
from cloudnetpy.instruments import ceilo2nc
from tests.unit.all_products_fun import Check
from tests.unit.lidar_fun import LidarFun

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
FILES = glob.glob(f"{SCRIPT_PATH}/data/da10/*.nc")
FILES.sort()

SITE_META = {
    "name": "Lindenberg",
    "altitude": 123,
    "latitude": 45.0,
    "longitude": 22.0,
    "model": "da10",
}


class TestDa10(Check):
    site_meta = SITE_META
    date = "2025-09-15"
    temp_dir = TemporaryDirectory()
    daily_file = temp_dir.name + "/daily.nc"
    concat_lib.concatenate_files(FILES, daily_file)
    temp_path = temp_dir.name + "/test.nc"
    uuid = ceilo2nc(daily_file, temp_path, site_meta, date=date)

    def test_common_lidar(self):
        lidar_fun = LidarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in LidarFun.__dict__.items():
            if "test_" in name:
                getattr(lidar_fun, name)()

    def test_variable_values(self):
        assert abs(self.nc.variables["wavelength"][:] - 911.0) < 0.001
        assert self.nc.variables["zenith_angle"][:].data == 1.65

    def test_comments(self):
        assert "SNR threshold applied: 5" in self.nc.variables["beta"].comment
