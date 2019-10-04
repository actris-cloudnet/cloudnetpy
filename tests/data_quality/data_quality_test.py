"""
For reliability testing of total CloudnetPy processing reference data is needed.
Reference data is statistical and momentary data from clear case and usage is to
test effects of new or modified method in development of CloudnetPy.
"""
import os
import glob
import pytest
import logging
import numpy as np
import numpy.ma as ma
import netCDF4
import configparser
from cloudnetpy.utils import calc_relative_error
from tests.test import LoggingHandler


def _get_root_path():
    path = f"{os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]}"
    return f"{os.path.split(path)[0]}/"


config = configparser.ConfigParser()
config.optionxform = str
config.read('ref_quantity_config.ini')

ROOT_PATH = _get_root_path()
REFERENCE_PATH = f"{ROOT_PATH}tests/reference_data/"
PROCESS_PATH = f"{ROOT_PATH}tests/source_data/"
LIMIT = 20

LOG = LoggingHandler()
logger = LOG.manage_logger_file()


def generate_data_testing(quantity, site, date, operative_path, reference=False, reprocess=False):
    logger.name = site
    #LoggingHandler.create_logger_file(PROCESS_PATH, site)
    LoggingHandler.logger_file_no_handler(PROCESS_PATH, site)
    LoggingHandler.logger_information(site, date)

    test_path = config[quantity]["path"]
    options = "--tb=line"
    if not reference:
        pytest.main([options, f"{ROOT_PATH}{test_path}", '-k', 'test_operative'])

    if reference:
        ref_test = pytest.main([options, f"{ROOT_PATH}{test_path}", '-k', 'test_reference'])
        # Even if test passes, get info about development statistics
    if reprocess and not reference:
        print("")
        # Reprocess reference datas if development is wanted


def get_process_data_and_boundaries(quantity):
    variables = config[quantity]['quantities']
    variables = list(variables.split(','))
    min_max_values = _parse_config_to_dict(variables)
    process_file_path = _get_test_file_name(PROCESS_PATH, quantity)
    results = {}
    for var in variables:
        process_data = netCDF4.Dataset(process_file_path)[var][:]
        minimum, maximum = min_max_values[var]
        results[var] = bool(minimum <= np.min(process_data*1000) and maximum >= np.max(process_data))
    return results


def get_reference_and_process_date(quantity):
    reference_variables = config[quantity]['quantities']
    reference_file_path = _get_test_file_name(REFERENCE_PATH, quantity)
    process_file_path = _get_test_file_name(PROCESS_PATH, quantity)
    reference_dict = {}
    process_dict = {}
    for var in reference_variables:
        reference_data = netCDF4.Dataset(reference_file_path)[var][:]
        process_data = netCDF4.Dataset(process_file_path)[var][:]
        reference_dict[var] = reference_data
        process_dict[var] = process_data
    return reference_dict, process_dict


def get_diff_of_data(ref_data, proc_data):
    differences = {}
    for key in ref_data.keys():
        diff = calc_relative_error(ref_data[key], proc_data[key])
        differences[key] = bool(-LIMIT < diff < LIMIT)
    return differences


def get_diff_of_median(ref_data, proc_data):
    differences = {}
    for key in ref_data.keys():
        diff = calc_relative_error(ma.median(ref_data[key]), ma.median(proc_data[key]))
        differences[key] = bool(-LIMIT < diff < LIMIT)
    return differences


def get_diff_of_mean(ref_data, proc_data):
    differences = {}
    for key in ref_data.keys():
        diff = calc_relative_error(np.mean(ref_data[key]), np.mean(proc_data[key]))
        differences[key] = bool(-LIMIT < diff < LIMIT)
    return differences


def get_diff_of_standard_dev(ref_data, proc_data):
    differences = {}
    for key in ref_data.keys():
        diff = calc_relative_error(np.std(ref_data[key]), np.std(proc_data[key]))
        differences[key] = bool(-LIMIT < diff < LIMIT)
    return differences


def get_number_of_pixels(ref_data, proc_data):
    differences = {}
    for key in ref_data.keys():
        n_true = np.sum(bool(ref_data[key]) == bool(proc_data[key]))
        diff = calc_relative_error(len(ref_data[key].ravel()), len(n_true.ravel()))
        differences[key] = bool(-LIMIT < diff < LIMIT)
    return differences

# Jakauma hommia voisi lisätä
# Tsekataan, ettei toisessa datassa ole voimakkaita piikkejä


def _get_test_file_name(path, quantity):
    files = glob.glob(f"{path}*.nc")
    for file in files:
        if quantity in file:
            return file


def _parse_config_to_dict(variables):
    dict = {}
    for var in variables:
        for q, val in config['min_max'].items():
            if var == q:
                dict[var] = tuple(map(int, val.split(',')))
    return dict


def false_variables_msg(results):
    key_type = 'Variable(s)'
    false_variables = []
    for var, val in results.items():
        if val is False:
            false_variables.append(var)
    return f"{key_type} {', '.join(false_variables)} not within boundaries!"


if __name__ == "__main__":
    operative_path = '/home/korpinen/ACTRIS/cloudnetpy/tests/source_data/'
    generate_data_testing('radar', 'Mace-Head', '20190703', operative_path)



