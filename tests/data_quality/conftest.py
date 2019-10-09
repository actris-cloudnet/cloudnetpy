import numpy as np
import pytest
import netCDF4
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
        self.wrong_value = {}
        self.value = False
        self._read_var_limits()

    def _read_var_limits(self):
        config = dict(DATA_CONFIG.items('limits'))
        nc = netCDF4.Dataset(self.file_name)
        keys = nc.variables.keys()
        for var, c_val in config.items():
            c_val = tuple(map(float, c_val.split(', ')))
            if var in keys:
                if c_val[0] > np.min(nc.variables[var][:]) or c_val[1] < np.max(nc.variables[var][:]):
                    self.value = True
                    self.wrong_value[var] = [np.min(nc.variables[var][:]), np.max(nc.variables[var][:])]
        nc.close()


def pytest_addoption(parser):
    parser.addoption('--test_file', action='store', help='Input file name')
