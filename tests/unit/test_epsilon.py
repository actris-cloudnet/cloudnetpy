import glob
from os import path
from tempfile import TemporaryDirectory
from pathlib import Path

import netCDF4

from cloudnetpy.products.epsilon import generate_epsilon_from_lidar
from tests.unit.all_products_fun import Check

SCRIPT_PATH = path.dirname(path.realpath(__file__))
data_dir = Path(f"{SCRIPT_PATH}/data/epsilon/")


class TestEpsilonLidar(Check):
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/test.nc"
    doppler_lidar_file = data_dir / "doppler-lidar.nc"
    doppler_lidar_wind_file = data_dir / "doppler-lidar-wind.nc"
    uuid = generate_epsilon_from_lidar(
        doppler_lidar_file, doppler_lidar_wind_file, temp_path, None
    )

    date = "2024-04-13"
    site_meta = {
        "name": "Jülich",
        "altitude": 111,
        "latitude": 50.908,
        "longitude": 6.413,
    }

    def test_default_processing(self, tmp_path):
        with netCDF4.Dataset(self.temp_path) as nc:
            assert nc["epsilon"][:].size == 39360
            assert 1e-4 < nc["epsilon"][:].mean() < 3e-4
            assert nc.cloudnet_file_type == "epsilon-lidar"
