import os
import sys
from tempfile import NamedTemporaryFile
import warnings
import netCDF4
from cloudnetpy.instruments import ceilo2nc, mira2nc
from cloudnetpy.categorize import generate_categorize
from all_products_fun import Check

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_PATH)


filepath = f"{SCRIPT_PATH}/../source_data/"


class TestCategorize(Check):
    date = "2021-11-20"
    site_meta = {"name": "Munich", "altitude": 538, "latitude": 48.5, "longitude": 11.5}
    radar_file = NamedTemporaryFile()
    lidar_file = NamedTemporaryFile()

    uuid_radar = mira2nc(f"{filepath}raw_mira_radar.mmclx", radar_file.name, site_meta)
    uuid_lidar = ceilo2nc(f"{filepath}raw_chm15k_lidar.nc", lidar_file.name, site_meta)

    input_files = {
        "radar": radar_file.name,
        "lidar": lidar_file.name,
        "mwr": f"{filepath}hatpro_mwr.nc",
        "model": f"{filepath}ecmwf_model.nc",
    }

    temp_file = NamedTemporaryFile()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        uuid = generate_categorize(input_files, temp_file.name)

    def test_global_attributes(self):
        nc = netCDF4.Dataset(self.temp_file.name)
        assert nc.title == "Cloud categorization products from Munich"
        nc.close()
