import pytest
import netCDF4
from tests.metadata_control import read_meta_config
from tests.utils import get_file_type

META_CONFIG = read_meta_config()


@pytest.fixture
def variable_names(pytestconfig):
    nc = netCDF4.Dataset(pytestconfig.option.test_file)
    file_type = get_file_type(pytestconfig.option.test_file)
    keys = set(nc.variables.keys())
    nc.close()
    try:
        missing = set(META_CONFIG['required_variables'][file_type].split(', ')) - keys
    except:
        missing = False
    return missing


@pytest.fixture
def global_attribute_names(pytestconfig):
    nc = netCDF4.Dataset(pytestconfig.option.test_file)
    file_type = get_file_type(pytestconfig.option.test_file)
    keys = set(nc.ncattrs())
    nc.close()
    print(keys)
    try:
        missing = set(META_CONFIG['required_attributes'][file_type].split(', ')) - keys
    except:
        missing = False
    return missing


@pytest.fixture
def global_attribute(pytestconfig):
    file = pytestconfig.option.test_file
    return GlobalAttribute(file)


@pytest.fixture
def variable(pytestconfig):
    file = pytestconfig.option.test_file
    return Variable(file)


class GlobalAttribute:
    def __init__(self, file):
        self.file_name = file
        self.wrong_unit = {}
        self.wrong_value = {}
        self.value = False
        self.unit = False
        self._read_attr_units()
        self._read_attr_limits()

    def _read_attr_units(self):
        config = dict(META_CONFIG.items('attributes_units'))
        nc = netCDF4.Dataset(self.file_name)
        keys = list(map(lambda x: x.lower(), nc.ncattrs()))
        print(self.file_name)
        for attr, c_unit in config.items():
            if attr in keys:
                print(attr)
                if not c_unit is getattr(nc.variables[attr], 'units', None):
                    self.unit = True
                    self.wrong_unit[attr] = getattr(nc.variables[attr], 'units', None)
        nc.close()

    def _read_attr_limits(self):
        config = dict(META_CONFIG.items('attributes_limits'))
        nc = netCDF4.Dataset(self.file_name)
        keys = nc.ncattrs()
        for attr, c_val in config.items():
            c_val = tuple(map(float, c_val.split(', ')))
            if attr in keys:
                if c_val[0] < min(nc.variables[attr][:]) or c_val[1] > max(nc.variables[attr][:]):
                    self.value = True
                    self.wrong_value[attr] = nc.variables[attr][:]
        nc.close()


class Variable:
    def __init__(self, pytestconfig):
        self.file_name = pytestconfig.option.test_file
        self.wrong_unit = {}
        self.wrong_value = {}
        self.value = False
        self.unit = False

    def _read_var_units(self):
        config = dict(META_CONFIG.items('variables_units'))
        nc = netCDF4.Dataset(self.file_name)
        keys = list(map(lambda x: x.lower(), nc.ncattrs()))
        print(self.file_name)
        for attr, c_unit in config.items():
            if attr in keys:
                print(attr)
                if not c_unit is getattr(nc.variables[attr], 'units', None):
                    self.unit = True
                    self.wrong_unit[attr] = getattr(nc.variables[attr], 'units', None)
        nc.close()

    def _read_var_limits(self):
        config = dict(META_CONFIG.items('variables_limits'))
        nc = netCDF4.Dataset(self.file_name)
        keys = nc.ncattrs()
        for attr, c_val in config.items():
            c_val = tuple(map(float, c_val.split(', ')))
            if attr in keys:
                if c_val[0] < min(nc.variables[attr][:]) or c_val[1] > max(nc.variables[attr][:]):
                    self.value = True
                    self.wrong_value[attr] = nc.variables[attr][:]
        nc.close()


def pytest_addoption(parser):
    parser.addoption('--test_file', action='store', help='Input file name')
