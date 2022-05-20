import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import importlib
from uuid import UUID

import netCDF4
from cloudnetpy_qc import Quality

from cloudnetpy.categorize import generate_categorize
from cloudnetpy.instruments import ceilo2nc, mira2nc


def _process_product_file(product_type: str, path: str, categorize_file: str) -> tuple:
    output_file = f"{path}{product_type}.nc"
    module = importlib.import_module(f"cloudnetpy.products")
    uuid = getattr(module, f"generate_{product_type}")(categorize_file, output_file)
    return output_file, uuid


def main():

    test_path = Path(__file__).parent
    source_path = f"{test_path}/source_data/"

    raw_files = {
        "radar": f"{source_path}raw_mira_radar.mmclx",
        "lidar": f"{source_path}raw_chm15k_lidar.nc",
    }

    calibrated_files = {"radar": f"{source_path}radar.nc", "lidar": f"{source_path}lidar.nc"}

    site_meta = {"name": "Munich", "altitude": 538, "latitude": 48.148, "longitude": 11.573}
    uuid_radar = mira2nc(raw_files["radar"], calibrated_files["radar"], site_meta, uuid="kissa")
    assert uuid_radar == "kissa"
    lidar_meta = site_meta.copy()
    uuid_lidar = ceilo2nc(raw_files["lidar"], calibrated_files["lidar"], lidar_meta)
    for _, filename in calibrated_files.items():
        _run_tests(filename)
    _check_attributes(calibrated_files["radar"], site_meta)
    _check_is_valid_uuid(uuid_lidar)

    input_files = {
        "radar": calibrated_files["radar"],
        "lidar": calibrated_files["lidar"],
        "mwr": f"{source_path}hatpro_mwr.nc",
        "model": f"{source_path}ecmwf_model.nc",
    }

    uuid_model, uuid_mwr = _get_uuids(input_files)
    categorize_file = f"{source_path}categorize.nc"
    uuid_categorize = generate_categorize(input_files, categorize_file)
    _run_tests(categorize_file)
    _check_is_valid_uuid(uuid_categorize)
    _check_source_file_uuids(categorize_file, (uuid_lidar, uuid_radar, uuid_model, uuid_mwr))
    product_file_types = ["classification", "iwc", "lwc", "drizzle", "der", "ier"]
    for file in product_file_types:
        product_file, uuid_product = _process_product_file(file, source_path, categorize_file)
        _run_tests(product_file)
        _check_is_valid_uuid(uuid_product)
        _check_attributes(product_file, site_meta)
        _check_source_file_uuids(product_file, (uuid_categorize,))


def _run_tests(filename: str):
    quality = Quality(filename)
    res_quality = quality.check_data()
    res_metadata = quality.check_metadata()
    assert quality.n_metadata_test_failures == 0, f"{filename} - {res_metadata}"
    assert quality.n_data_test_failures == 0, f"{filename} - {res_quality}"


def _check_source_file_uuids(file: str, expected_uuids: tuple):
    with netCDF4.Dataset(file) as nc:
        source_uuids = nc.source_file_uuids.replace(",", "").split(" ")
        for uuid in expected_uuids:
            assert uuid in source_uuids
        for uuid in source_uuids:
            assert uuid in expected_uuids


def _check_is_valid_uuid(uuid):
    try:
        UUID(uuid, version=4)
    except (ValueError, TypeError):
        raise AssertionError(f"{uuid} is not a valid UUID.")


def _check_attributes(full_path: str, metadata: dict):
    with netCDF4.Dataset(full_path) as nc:
        assert nc.variables["altitude"][:] == metadata["altitude"]


def _get_uuids(data: dict) -> tuple:
    with netCDF4.Dataset(data["model"]) as nc:
        uuid_model = nc.file_uuid
    with netCDF4.Dataset(data["mwr"]) as nc:
        uuid_mwr = nc.file_uuid
    return uuid_model, uuid_mwr


if __name__ == "__main__":
    main()
