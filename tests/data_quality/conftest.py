import numpy as np
import pytest
import netCDF4
from tests.utils import read_config, find_missing_keys

CONFIG = read_config('data_quality/data_quality_config.ini')


@pytest.fixture
def missing_variables(pytestconfig):
    file_name = pytestconfig.option.test_file
    return find_missing_keys(CONFIG, 'quantities', file_name)


@pytest.fixture
def data(pytestconfig):
    file = pytestconfig.option.test_file
    return InputData(file)


class InputData:
    def __init__(self, file):
        self.file_name = file
        self.bad_values = {}
        self.value = False
        self._check_min_max_values()

    def _check_min_max_values(self):
        nc = netCDF4.Dataset(self.file_name)
        keys = nc.variables.keys()
        config = CONFIG.items('limits')
        for var, limits in config:
            if var in keys:
                limits = tuple(map(float, limits.split(',')))
                min_value = np.min(nc.variables[var][:])
                max_value = np.max(nc.variables[var][:])
                if min_value < limits[0] or max_value > limits[1]:
                    self.value = True
                    self.bad_values[var] = [min_value, max_value]
        nc.close()


def pytest_addoption(parser):
    parser.addoption('--test_file', action='store', help='Input file name')
