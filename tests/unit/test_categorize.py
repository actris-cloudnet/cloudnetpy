import os
import shutil
from tempfile import TemporaryDirectory

import netCDF4

from cloudnetpy.categorize import generate_categorize, CategorizeInput
from cloudnetpy.instruments import ceilo2nc, mira2nc
from tests.unit.all_products_fun import Check

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
filepath = f"{SCRIPT_PATH}/../source_data"


class TestCategorize(Check):
    date = "2021-11-20"
    site_meta = {
        "name": "Munich",
        "altitude": 538,
        "latitude": 48.5,
        "longitude": 11.5,
    }

    temp_dir = TemporaryDirectory()
    radar_path = temp_dir.name + "/radar.nc"
    lidar_path = temp_dir.name + "/lidar.nc"

    uuid_radar = mira2nc(f"{filepath}/raw_mira_radar.mmclx", radar_path, site_meta)
    uuid_lidar = ceilo2nc(f"{filepath}/raw_chm15k_lidar.nc", lidar_path, site_meta)

    input_files: CategorizeInput = {
        "radar": radar_path,
        "lidar": lidar_path,
        "mwr": f"{filepath}/hatpro_mwr.nc",
        "model": f"{filepath}/ecmwf_model.nc",
    }

    temp_path = temp_dir.name + "/categorize.nc"
    uuid = generate_categorize(input_files, temp_path)

    def test_global_attributes(self):
        with netCDF4.Dataset(self.temp_path) as nc:
            assert nc.title == "Cloud categorization products from Munich"

    def test_categorize_without_lwp(self):
        input_files_without_mwr = self.input_files.copy()
        del input_files_without_mwr["mwr"]
        temp_path = self.temp_dir.name + "/categorize_without_lwp.nc"
        generate_categorize(input_files_without_mwr, temp_path)

    def test_zh_offset_not_in_output_by_default(self):
        with netCDF4.Dataset(self.temp_path) as nc:
            assert "Z_offset" not in nc.variables
            assert (
                nc.variables["Z"].ancillary_variables == "Z_error Z_bias Z_sensitivity"
            )

    def test_zh_offset_passed_through(self):
        radar_path = self.temp_dir.name + "/radar_with_offset.nc"
        shutil.copy(self.radar_path, radar_path)
        with netCDF4.Dataset(radar_path, "a") as nc:
            var = nc.createVariable("Zh_offset", "f4")
            var[:] = 2.5
        input_files = self.input_files.copy()
        input_files["radar"] = radar_path
        temp_path = self.temp_dir.name + "/categorize_with_offset.nc"
        generate_categorize(input_files, temp_path)
        with netCDF4.Dataset(temp_path) as nc:
            var = nc.variables["Z_offset"]
            assert var[:] == 2.5
            assert var.dimensions == ()
            assert var.units == "dBZ"
            assert var.source == nc.variables["Z"].source
            assert nc.variables["Z"].ancillary_variables.endswith(" Z_offset")
