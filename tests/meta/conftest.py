import numpy as np
import pytest
import netCDF4
from tests.utils import read_config, find_missing_keys

CONFIG = read_config('meta/metadata_config.ini')


@pytest.fixture
def missing_variables(pytestconfig):
    file_name = pytestconfig.option.test_file
    return find_missing_keys(CONFIG, 'required_variables', file_name)


@pytest.fixture
def missing_global_attributes(pytestconfig):
    file_name = pytestconfig.option.test_file
    return find_missing_keys(CONFIG, 'required_attributes', file_name)


@pytest.fixture
def global_attribute(pytestconfig):
    return GlobalAttribute(pytestconfig.option.test_file)


@pytest.fixture
def variable(pytestconfig):
    return Variable(pytestconfig.option.test_file)


class GlobalAttribute:
    def __init__(self, file):
        self.file_name = file
        self.wrong_units = {}
        self.wrong_values = {}
        self.value = False
        self.unit = False
        self._read_attr_units()
        self._read_attr_limits()

    def _read_attr_units(self):
        nc = netCDF4.Dataset(self.file_name)
        keys = nc.ncattrs()
        config = CONFIG.items('attributes_units')
        for attr, reference in config:
            if attr in keys:
                value = nc.getncattr(attr)
                if reference != value:
                    self.unit = True
                    self.wrong_units[attr] = value
        nc.close()

    def _read_attr_limits(self):
        nc = netCDF4.Dataset(self.file_name)
        keys = nc.ncattrs()
        config = CONFIG.items('attributes_limits')
        for attr, limits in config:
            limits = tuple(map(float, limits.split(',')))
            if attr in keys:
                value = int(nc.getncattr(attr))
                if value < limits[0] or value > limits[1]:
                    self.value = True
                    self.wrong_values[attr] = value
        nc.close()


class Variable:
    def __init__(self, file):
        self.file_name = file
        self.wrong_units = {}
        self.wrong_values = {}
        self.value = False
        self.unit = False
        self._read_var_units()
        self._read_var_limits()

    def _read_var_units(self):
        nc = netCDF4.Dataset(self.file_name)
        keys = nc.variables.keys()
        config = CONFIG.items('variables_units')
        for var, reference in config:
            if var in keys:
                value = nc.variables[var].units
                if reference != value:
                    self.unit = True
                    self.wrong_units[var] = value
        nc.close()

    def _read_var_limits(self):
        nc = netCDF4.Dataset(self.file_name)
        keys = nc.variables.keys()
        config = CONFIG.items('variables_limits')
        for var, limits in config:
            limits = tuple(map(float, limits.split(',')))
            if var in keys:
                min_value = np.min(nc.variables[var][:])
                max_value = np.max(nc.variables[var][:])
                if min_value < limits[0] or max_value > limits[1]:
                    self.value = True
                    self.wrong_values[var] = [min_value, max_value]
        nc.close()


def pytest_addoption(parser):
    parser.addoption('--test_file', action='store', help='Input file name')
