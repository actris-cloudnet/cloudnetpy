import os
import sys
sys.path.insert(0, os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
import warnings
import logging
import glob
from zipfile import ZipFile
import pytest
import requests
import netCDF4
from tests import run_testcase_processing as process
from tests.test_tools import remove_import_modules

warnings.filterwarnings("ignore")
OPERATIVE_RUN = True


class LoggingHandler:
    @staticmethod
    def manage_logger_file():
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        return logger

    @staticmethod
    def create_logger_file(path, name):
        fh = logging.FileHandler(os.path.join(path, f"{name}.log"), mode='w')
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    @staticmethod
    def logger_file_no_handler(path, site):
        logging.basicConfig(filename=f'{path}{site}.log',
                            filemode='w',
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.DEBUG)

    @staticmethod
    def fill_log(reason, instru):
        if OPERATIVE_RUN:
            logger.warning(f"Data quality test didn't pass in {instru} file")
            logger.warning(reason)

    @staticmethod
    def logger_information(site, date):
        logger.info(f"Operative processing from {site} at {date}")


LOG = LoggingHandler()
logger = LOG.manage_logger_file()


def get_default_path():
    # TODO: Change path type to be operative or reference
    if OPERATIVE_RUN:
        # Operative process file path here
        return f"{os.path.dirname(os.path.abspath(__file__))}/source_data/"
    else:
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


def main(site, date, reference_run=False):
    c_path = f"{os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]}/cloudnetpy/"
    options = "--tb=line"

    if not reference_run:
        path = get_default_path()
        logger.name = site
        LoggingHandler.create_logger_file(path, site)
        LoggingHandler.logger_information(site, date)

        print(f"\n{22 * '#'} Running operative CloudnetPy tests {22 * '#'}")
        print("\nTesting raw files:\n")
        pytest.main([options, f"{c_path}instruments/tests/raw_files_test.py"])
        remove_import_modules()
        # Here calling for data

        print("\nTesting calibrated files:\n")
        pytest.main([options, f"{c_path}instruments/tests/calibrated_files_test.py"])

        print("\nTesting categorize file:\n")
        pytest.main([options, f"{c_path}categorize/tests/categorize_file_test.py"])

        print("\nTesting product files:\n")
        pytest.main([options, f"{c_path}products/tests/product_files_test.py"])

    else:
        global OPERATIVE_RUN
        OPERATIVE_RUN = False

        print(f"\n{22*'#'} Running all CloudnetPy tests {22*'#'}")
        input_path = get_default_path()
        _load_test_data(input_path)

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
        remove_import_modules()

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


def read_attribute(identifier, name):
    file = get_test_file_name(identifier)
    return getattr(netCDF4.Dataset(file), name)


def read_variable(identifier, name):
    file = get_test_file_name(identifier)
    return netCDF4.Dataset(file).variables[name]


def _check_failures(tests, var):
    if tests in (1, 3):
        print(f"\n{20*'#'} Error in {var} file testing! {20*'#'}")
        sys.exit()


def missing_key_msg(missing_keys, name, is_attr=False):
    key_type = 'Attribute(s)' if is_attr else 'Variable(s)'
    return f"{key_type} {missing_keys} missing in {name} file!"


def bad_value_msg(name, value):
    return f"Error in value of {name}: {value}"


if __name__ == "__main__":
    main()
