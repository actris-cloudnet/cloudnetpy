import os
import sys
sys.path.insert(0, os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
import warnings
import glob
from zipfile import ZipFile
import pytest
import requests
import netCDF4
from tests import run_testcase_processing as process

warnings.filterwarnings("ignore")


def get_default_path():
    return f"{os.path.dirname(os.path.abspath(__file__))}/source_data/"


def _load_test_data(input_path):
    def _extract_zip():
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
        _extract_zip()
    else:
        is_file = os.path.isfile(full_zip_name)
        if not is_file:
            _extract_zip()


def main():

    print(f"\n{22*'#'} Running all CloudnetPy tests {22*'#'}")

    c_path = f"{os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]}/cloudnetpy/"
    input_path = get_default_path()
    _load_test_data(input_path)

    options = "--tb=line"
    site = 'mace-head'

    print("\nTesting misc CloudnetPy routines:\n")
    test = pytest.main([options, f"{c_path}tests/test_utils.py"])
    _check_failures(test, "utils.py")
    test = pytest.main([options, f"{c_path}categorize/tests/test_atmos.py"])
    _check_failures(test, "atmos.py")
    test = pytest.main([options, f"{c_path}categorize/tests/test_classify.py"])
    _check_failures(test, "classify.py")

    print("\nTesting raw files:\n")
    test = pytest.main([options, f"{c_path}instruments/tests/raw_files_test.py"])
    _check_failures(test, "raw")

    print("\nProcessing CloudnetPy calibrated files from raw files:\n")
    process.process_cloudnetpy_raw_files(site, input_path)

    print("\nTesting calibrated files:\n")
    test = pytest.main([options, f"{c_path}instruments/tests/calibrated_files_test.py"])
    _check_failures(test, "calibrated")

    print("\nProcessing CloudnetPy categorize file:\n")
    process.process_cloudnetpy_categorize(site, input_path)

    print("\nTesting categorize file:\n")
    test = pytest.main([options, f"{c_path}categorize/tests/categorize_file_test.py"])
    _check_failures(test, "categorize")

    print("\nProcessing CloudnetPy product files:\n")
    process.process_cloudnetpy_products(input_path)

    print("\nTesting product files:\n")
    test = pytest.main([options, f"{c_path}products/tests/product_files_test.py"])
    _check_failures(test, "product")

    print(f"\n{10*'#'} All tests passed and processing works correctly! {10*'#'}")


def get_test_file_name(identifier):
    path = get_default_path()
    files = glob.glob(f"{path}*.nc")
    for file in files:
        if identifier in file:
            return file
    raise FileNotFoundError


def read_variable_names(identifier):
    file = get_test_file_name(identifier)
    return set(netCDF4.Dataset(file).variables.keys())


def read_attribute_names(identifier):
    file = get_test_file_name(identifier)
    return set(netCDF4.Dataset(file).ncattrs())


def _check_failures(tests, var):
    if tests in (1, 3):
        print(f"\n{20*'#'} Error in {var} file testing! {20*'#'}")
        sys.exit()


def missing_var_msg(missing_keys, name):
    return f"Variable(s) {missing_keys} missing in {name} file!"


def missing_attr_msg(missing_keys, name):
    return f"Attribute(s) {missing_keys} missing in {name} file!"


if __name__ == "__main__":
    main()
