import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zipfile import ZipFile
import requests
import importlib
from tests.utils import get_test_path
from cloudnetpy.instruments import mira, ceilo
from cloudnetpy.categorize import categorize
from tests import api

PROCESS = False


def _load_test_data(input_path):

    def _load_zip():
        sys.stdout.write("\nLoading input files...")
        r = requests.get(url)
        open(full_zip_name, 'wb').write(r.content)
        fl = ZipFile(full_zip_name, 'r')
        fl.extractall(input_path)
        fl.close()
        sys.stdout.write("    Done.\n")

    url = 'http://devcloudnet.fmi.fi/files/cloudnetpy_test_input_files.zip'
    zip_name = os.path.split(url)[-1]
    full_zip_name = f"{input_path}{zip_name}"
    is_dir = os.path.isdir(input_path)
    if not is_dir:
        os.mkdir(input_path)
        _load_zip()
    else:
        is_file = os.path.isfile(full_zip_name)
        if not is_file:
            _load_zip()


def _process_product_file(product_type, path, categorize_file):
    output_file = f"{path}{product_type}.nc"
    module = importlib.import_module(f"cloudnetpy.products.{product_type}")
    if PROCESS:
        getattr(module, f"generate_{product_type}")(categorize_file, output_file)
    return output_file


def main():

    test_path = get_test_path()
    source_path = f"{test_path}/source_data/"
    _load_test_data(source_path)
    prefix = '20190517_mace-head_'
    log_file = f"{source_path}Mace_Head.log"

    raw_files = {
        'radar': f"{source_path}{prefix}mira_raw.nc",
        'lidar': f"{source_path}{prefix}chm15k_raw.nc",
    }
    for name, file in raw_files.items():
        api.check_metadata(file, log_file)

    calibrated_files = {
        'radar': f"{source_path}radar.nc",
        'lidar': f"{source_path}lidar.nc",
    }
    site_meta = {'name': 'Mace Head', 'altitude': 13}
    if PROCESS:
        mira.mira2nc(raw_files['radar'], calibrated_files['radar'], site_meta)
        ceilo.ceilo2nc(raw_files['lidar'], calibrated_files['lidar'], site_meta)
    for name, file in calibrated_files.items():
        api.check_metadata(file, log_file)
        api.check_data_quality(file, log_file)

    input_files = {
        'radar': calibrated_files['radar'],
        'lidar': calibrated_files['lidar'],
        'mwr': f"{source_path}{prefix}hatpro.nc",
        'model': f"{source_path}{prefix}ecmwf.nc",
    }
    categorize_file = f"{source_path}categorize.nc"
    if PROCESS:
        categorize.generate_categorize(input_files, categorize_file)
    api.check_metadata(categorize_file, log_file)
    api.check_data_quality(categorize_file, log_file)

    product_file_types = ['iwc', 'lwc', 'drizzle', 'classification']
    for file in product_file_types:
        product_file = _process_product_file(file, source_path, categorize_file)
        api.check_metadata(product_file, log_file)
        api.check_data_quality(product_file, log_file)


if __name__ == "__main__":
    main()
