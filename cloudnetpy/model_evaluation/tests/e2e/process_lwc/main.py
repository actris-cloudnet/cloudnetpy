#!/usr/bin/env python3
import argparse
import os
import subprocess
from os import path
from tempfile import TemporaryDirectory

from cloudnetpy.model_evaluation.products import product_resampling

ROOT_PATH = os.path.abspath(os.curdir)
SCRIPT_PATH = path.dirname(path.realpath(__file__))
test_file_model = (
    f"{ROOT_PATH}/cloudnetpy/model_evaluation/tests/data/20190517_mace-head_ecmwf.nc"
)
test_file_product = f"{ROOT_PATH}/cloudnetpy/model_evaluation/tests/data/20190517_mace-head_lwc-scaled-adiabatic.nc"


def _process() -> None:
    tmp_dir = TemporaryDirectory()
    temp_file = f"{tmp_dir.name}/xx.nc"
    product_resampling.process_L3_day_product(
        "ecmwf",
        "lwc",
        [test_file_model],
        test_file_product,
        temp_file,
    )
    try:
        subprocess.call(
            [
                "pytest",
                "-v",
                f"{SCRIPT_PATH}/tests.py",
                "--full_path",
                temp_file,
            ],
        )
    except subprocess.CalledProcessError:
        raise
    tmp_dir.cleanup()


def main() -> None:
    _process()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model evaluation liquid water content processing e2e test.",
    )
    ARGS = parser.parse_args()
    main()
