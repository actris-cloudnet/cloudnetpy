import numpy as np
import pytest
import netCDF4
import logging
from tests.utils import get_file_type, read_data_config

DATA_CONFIG = read_data_config()


@pytest.fixture
def variable_names(pytestconfig):
    nc = netCDF4.Dataset(pytestconfig.option.test_file)
    file_type = get_file_type(pytestconfig.option.test_file)
    keys = set(nc.variables.keys())
    nc.close()
    try:
        missing = set(DATA_CONFIG[file_type]['quantities'].split(', ')) - keys
    except:
        missing = False
    return missing


@pytest.fixture
def data(pytestconfig):
    file = pytestconfig.option.test_file
    return InputData(file)


class InputData:
    def __init__(self, file):
        self.file_name = file
        self.bad_values = {}
        self.value = False
        self._read_var_limits()

    def _read_var_limits(self):
        nc = netCDF4.Dataset(self.file_name)
        keys = nc.variables.keys()
        config = DATA_CONFIG.items('limits')
        for var, limits in config:
            if var in keys:
                limits = tuple(map(float, limits.split(',')))
                min_value = np.min(nc.variables[var][:])
                max_value = np.max(nc.variables[var][:])
                if limits[0] > min_value or limits[1] < max_value:
                    self.value = True
                    self.bad_values[var] = [min_value, max_value]
        nc.close()


def pytest_addoption(parser):
    parser.addoption('--test_file', action='store', help='Input file name')
