import importlib
from pathlib import Path
import numpy as np

import netCDF4
from cloudnetpy_qc import quality

from cloudnetpy.categorize import generate_categorize, CategorizeInput
from cloudnetpy.instruments import ceilo2nc, mira2nc


def _process_product_file(product_type: str, path: str, categorize_file: str) -> tuple:
    output_file = f"{path}{product_type}.nc"
    module = importlib.import_module("cloudnetpy.products")
    uuid = getattr(module, f"generate_{product_type}")(categorize_file, output_file)
    return output_file, uuid


def main():
    test_path = Path(__file__).parent
    source_path = f"{test_path}/source_data/"

    raw_files = {
        "radar": f"{source_path}raw_mira_radar.mmclx",
        "lidar": f"{source_path}raw_chm15k_lidar.nc",
    }

    calibrated_files = {
        "radar": f"{source_path}radar.nc",
        "lidar": f"{source_path}lidar.nc",
    }

    site_meta = {
        "name": "Munich",
        "altitude": 538,
        "latitude": 48.148,
        "longitude": 11.573,
    }
    uuid_radar = mira2nc(
        raw_files["radar"],
        calibrated_files["radar"],
        site_meta,
        uuid="7e7e1d51-daea-4f9b-a1b3-103bd8ea3ce6",
    )
    assert str(uuid_radar) == "7e7e1d51-daea-4f9b-a1b3-103bd8ea3ce6"
    lidar_meta = site_meta.copy()
    uuid_lidar = ceilo2nc(raw_files["lidar"], calibrated_files["lidar"], lidar_meta)
    for _, filename in calibrated_files.items():
        _run_tests(filename)
    _check_attributes(calibrated_files["radar"], site_meta)

    input_files: CategorizeInput = {
        "radar": calibrated_files["radar"],
        "lidar": calibrated_files["lidar"],
        "mwr": f"{source_path}hatpro_mwr.nc",
        "model": f"{source_path}ecmwf_model.nc",
    }

    uuid_model, uuid_mwr = _get_uuids(input_files)
    categorize_file = f"{source_path}categorize.nc"
    uuid_categorize = generate_categorize(input_files, categorize_file)
    _run_tests(categorize_file)
    _check_source_file_uuids(
        categorize_file,
        (str(uuid_lidar), str(uuid_radar), uuid_model, uuid_mwr),
    )
    product_file_types = [
        "classification",
        "iwc",
        "lwc",
        "drizzle",
        "der",
        "ier",
    ]
    for file in product_file_types:
        product_file, uuid_product = _process_product_file(
            file,
            source_path,
            categorize_file,
        )
        _run_tests(product_file)
        _check_attributes(product_file, site_meta)
        _check_source_file_uuids(product_file, (str(uuid_categorize),))


def _run_tests(filename: str):
    n = 0
    report = quality.run_tests(
        Path(filename),
        {"time": None, "latitude": 0, "longitude": 0, "altitude": 0},
        ignore_tests=["TestCFConvention", "TestCoordinates"],
    )
    keys = ("TestUnits", "TestLongNames", "TestStandardNames")
    for test in report.tests:
        if test.test_id in keys:
            assert not test.exceptions, test.exceptions
            n += 1
    assert n == len(keys)


def _check_source_file_uuids(file: str, expected_uuids: tuple[str, ...]):
    with netCDF4.Dataset(file) as nc:
        source_uuids = nc.source_file_uuids.replace(",", "").split(" ")
        for uuid in expected_uuids:
            assert uuid in source_uuids
        for uuid in source_uuids:
            assert uuid in expected_uuids


def _check_attributes(full_path: str, metadata: dict):
    with netCDF4.Dataset(full_path) as nc:
        assert np.all(nc.variables["altitude"][:] == metadata["altitude"])


def _get_uuids(data: CategorizeInput) -> tuple[str, str]:
    with netCDF4.Dataset(data["model"]) as nc:
        uuid_model = nc.file_uuid
    with netCDF4.Dataset(data["mwr"]) as nc:
        uuid_mwr = nc.file_uuid
    return uuid_model, uuid_mwr


if __name__ == "__main__":
    main()
