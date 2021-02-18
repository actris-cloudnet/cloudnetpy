import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zipfile import ZipFile
import requests
import importlib
from cloudnetpy.instruments import mira2nc
from cloudnetpy.instruments import ceilo2nc
from cloudnetpy.categorize import generate_categorize
from tests import check_data_quality, check_metadata, utils
from tests import api


def _load_test_data(input_path: str):

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


def _process_product_file(product_type: str, path: str, categorize_file: str) -> tuple:
    output_file = f"{path}{product_type}.nc"
    module = importlib.import_module(f"cloudnetpy.products")
    uuid = getattr(module, f"generate_{product_type}")(categorize_file, output_file)
    return output_file, uuid


def main():

    test_path = utils.get_test_path()
    source_path = f"{test_path}/source_data/"
    _load_test_data(source_path)
    log_file = f"{source_path}Mace_Head.log"

    raw_files = {
        'radar': f"{source_path}raw_mira_radar.mmclx",
        'lidar': f"{source_path}raw_chm15k_lidar.nc",
    }
    """"
    We know these fail at the moment:
    for name, file in raw_files.items():
        check_metadata(file, log_file)
    """

    calibrated_files = {
        'radar': f"{source_path}radar.nc",
        'lidar': f"{source_path}lidar.nc",
    }
    site_meta = {'name': 'Mace Head', 'altitude': 13.0}
    uuid_radar = mira2nc(raw_files['radar'], calibrated_files['radar'], site_meta, uuid='kissa')
    assert uuid_radar == 'kissa'
    uuid_lidar = ceilo2nc(raw_files['lidar'], calibrated_files['lidar'], site_meta)
    for name, file in calibrated_files.items():
        check_metadata(file, log_file)
        check_data_quality(file, log_file)

    api.check_attributes(calibrated_files['radar'], site_meta)
    api.check_is_valid_uuid(uuid_lidar)

    input_files = {
        'radar': calibrated_files['radar'],
        'lidar': calibrated_files['lidar'],
        'mwr': f"{source_path}hatpro_mwr.nc",
        'model': f"{source_path}ecmwf_model.nc",
    }
    categorize_file = f"{source_path}categorize.nc"
    uuid_categorize = generate_categorize(input_files, categorize_file)
    check_metadata(categorize_file, log_file)
    check_data_quality(categorize_file, log_file)
    api.check_is_valid_uuid(uuid_categorize)
    api.check_source_file_uuids(categorize_file, (uuid_lidar, uuid_radar))

    product_file_types = ['classification', 'iwc', 'lwc', 'drizzle']
    for file in product_file_types:
        product_file, uuid_product = _process_product_file(file, source_path, categorize_file)
        check_metadata(product_file, log_file)
        check_data_quality(product_file, log_file)
        api.check_is_valid_uuid(uuid_product)
        api.check_attributes(product_file, site_meta)
        api.check_source_file_uuids(product_file, (uuid_categorize,))


if __name__ == "__main__":
    main()
