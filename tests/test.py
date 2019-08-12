import os, sys
import pytest
import glob
from zipfile import ZipFile
import requests
from tests.run_testcase_processing import *
import warnings

warnings.filterwarnings("ignore")
pytest_modules = ["pytest_remotedata", "pytest_openfiles", "pytest_doctestplus",
                  "pytest_arraydiff"]


def main():

    def _load_test_data():
        def _extract_zip(extract_path):
            sys.stdout.write("\nLoad test data...")
            r = requests.get(url)
            open(input_path + save_name, 'wb').write(r.content)
            fl = ZipFile(input_path + save_name, 'r')
            fl.extractall(extract_path)
            fl.close()
            sys.stdout.write("    Done.\n")

        save_name = os.path.split(url)[-1]
        is_dir = os.path.isdir(input_path)
        if not is_dir:
            os.mkdir(input_path )
            _extract_zip(input_path)
        else:
            is_file = os.path.isfile(input_path + save_name)
            if not is_file:
                _extract_zip(input_path)

    print("###################### Start testing CloudnetPy test case #######################")

    c_path = os.path.split(os.getcwd())[0]
    input_path = os.path.join(os.getcwd() + '/source_data/')
    url = 'http://devcloudnet.fmi.fi/files/cloudnetpy_test_input_files.zip'
    _load_test_data()

    print('\nTest raw files\n')
    test = pytest.main(["--tb=line", c_path + '/cloudnetpy/instruments/tests/raw_files_test.py'])
    _check_failures(test, "raw")
    _remove_import_modules(pytest_modules)

    print("\nProcessing CloudnetPy calibrated files from raw files")
    process_cloudnetpy_raw_files('mace-head', input_path)

    print('\nTest calibrated files\n')
    test = pytest.main(["--tb=line", c_path + '/cloudnetpy/instruments/tests/calibrated_files_test.py'])
    _check_failures(test, "calibrated")

    print("\nProcessing CloudnetPy categorize file from calibrated files")
    process_cloudnetpy_categorize('mace-head', input_path)

    print('\nTest category file\n')
    test = pytest.main(["--tb=line", c_path + '/cloudnetpy/categorize/tests/categorize_file_test.py'])
    _check_failures(test, "category")

    print("\nProcessing CloudnetPy product files from categorize file")
    process_cloudnetpy_products('mace-head', input_path)

    print('\nTest product files\n')
    test = pytest.main(["--tb=line", c_path + '/cloudnetpy/products/tests/product_files_test.py'])
    _check_failures(test, "product")

    print("\n########## All tests passed and processing cloudnetPy done correctly ###########")


def initialize_test_data(instrument, source_path=None):
    """
    Finds all file paths and parses wanted files to list
    """
    if not source_path:
        source_path = os.getcwd() + '/source_data/'
    test_data = glob.glob(source_path + '*.nc')
    paths = []
    for inst in instrument:
        for file in test_data:
            if inst in file:
              paths.append(file)
    return paths


def _remove_import_modules(pytest_modules):
    for module in pytest_modules:
        if module in sys.modules:
            del sys.modules[module]


def _check_failures(tests, var):
    if tests == 1 or tests == 3:
        print("\n"
              "####################"
              f"# Failures in processing {var} files #"
              "####################")
        sys.exit()


if __name__ == "__main__":
    main()
