import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zipfile import ZipFile
from uuid import UUID
import requests
import importlib
from cloudnetpy.instruments import mira2nc
from cloudnetpy.instruments import ceilo2nc
from cloudnetpy.categorize import generate_categorize
from cloudnetpy.quality import Quality
import netCDF4


def _load_test_data(input_path: str):

    def _load_zip():
        sys.stdout.write("\nLoading input files...")
        r = requests.get(url)
        open(full_zip_name, 'wb').write(r.content)
        fl = ZipFile(full_zip_name, 'r')
        fl.extractall(input_path)
        fl.close()
        sys.stdout.write("    Done.\n")

    url = 'https://lake.fmi.fi/cloudnet-public/cloudnetpy_test_input_files.zip'
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

    test_path = Path(__file__).parent
    source_path = f"{test_path}/source_data/"
    _load_test_data(source_path)

    raw_files = {
        'radar': f"{source_path}raw_mira_radar.mmclx",
        'lidar': f"{source_path}raw_chm15k_lidar.nc"}

    calibrated_files = {
        'radar': f"{source_path}radar.nc",
        'lidar': f"{source_path}lidar.nc"}

    site_meta = {'name': 'Mace Head', 'altitude': 13.0}
    uuid_radar = mira2nc(raw_files['radar'], calibrated_files['radar'], site_meta, uuid='kissa')
    assert uuid_radar == 'kissa'
    uuid_lidar = ceilo2nc(raw_files['lidar'], calibrated_files['lidar'], site_meta)
    for _, filename in calibrated_files.items():
        _run_tests(filename)
    _check_attributes(calibrated_files['radar'], site_meta)
    _check_is_valid_uuid(uuid_lidar)

    input_files = {
        'radar': calibrated_files['radar'],
        'lidar': calibrated_files['lidar'],
        'mwr': f"{source_path}hatpro_mwr.nc",
        'model': f"{source_path}ecmwf_model.nc"}

    categorize_file = f"{source_path}categorize.nc"
    uuid_categorize = generate_categorize(input_files, categorize_file)
    _run_tests(categorize_file)
    _check_is_valid_uuid(uuid_categorize)
    _check_source_file_uuids(categorize_file, (uuid_lidar, uuid_radar))

    product_file_types = ['classification', 'iwc', 'lwc', 'drizzle']
    for file in product_file_types:
        product_file, uuid_product = _process_product_file(file, source_path, categorize_file)
        _run_tests(product_file)
        _check_is_valid_uuid(uuid_product)
        _check_attributes(product_file, site_meta)
        _check_source_file_uuids(product_file, (uuid_categorize,))


def _run_tests(filename: str):
    quality = Quality(filename)
    quality.check_data()
    quality.check_metadata()
    assert quality.n_metadata_test_failures == 0
    assert quality.n_data_test_failures == 0


def _check_source_file_uuids(file: str, expected_uuids: tuple):
    nc = netCDF4.Dataset(file)
    source_uuids = nc.source_file_uuids.replace(',', '').split(' ')
    for uuid in expected_uuids:
        assert uuid in source_uuids
    for uuid in source_uuids:
        assert uuid in expected_uuids
    nc.close()


def _check_is_valid_uuid(uuid):
    try:
        UUID(uuid, version=4)
    except (ValueError, TypeError):
        raise AssertionError(f'{uuid} is not a valid UUID.')


def _check_attributes(full_path: str, metadata: dict):
    nc = netCDF4.Dataset(full_path)
    assert nc.variables['altitude'][:] == metadata['altitude']
    nc.close()


if __name__ == "__main__":
    main()
