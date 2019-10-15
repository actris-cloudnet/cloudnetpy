import pytest
import netCDF4
import numpy as np
from tests import utils

CONFIG = utils.read_config('meta/metadata_config.ini')


@pytest.fixture
def missing_variables(pytestconfig):
    file_name = pytestconfig.option.test_file
    return _find_missing_keys('required_variables', file_name)


@pytest.fixture
def missing_global_attributes(pytestconfig):
    file_name = pytestconfig.option.test_file
    return _find_missing_keys('required_global_attributes', file_name)


@pytest.fixture
def global_attribute(pytestconfig):
    return GlobalAttribute(pytestconfig.option.test_file)


@pytest.fixture
def variable(pytestconfig):
    return Variable(pytestconfig.option.test_file)


class GlobalAttribute:
    def __init__(self, file):
        self.file_name = file
        self.bad_values = self._check_values()

    def _check_values(self):
        bad = {}
        nc = netCDF4.Dataset(self.file_name)
        nc_keys = nc.ncattrs()
        for attr, limits in CONFIG.items('attribute_limits'):
            if attr in nc_keys:
                limits = tuple(map(float, limits.split(',')))
                value = int(nc.getncattr(attr))
                if not limits[0] <= value <= limits[1]:
                    bad[attr] = value
        nc.close()
        return bad


class Variable:
    def __init__(self, file):
        self.file_name = file
        self.bad_units = self._check_units()

    def _check_units(self):
        bad = {}
        nc = netCDF4.Dataset(self.file_name)
        nc_keys = nc.variables.keys()
        for var, reference in CONFIG.items('variable_units'):
            if var in nc_keys:
                value = nc.variables[var].units
                if reference != value:
                    bad[var] = value
        nc.close()
        return bad


def _find_missing_keys(config_field, file_name):
    nc = netCDF4.Dataset(file_name)
    nc_keys = _read_nc_keys(nc, config_field)
    nc.close()
    try:
        config_keys = _read_config_keys(config_field, file_name)
    except KeyError:
        return False
    return set(config_keys) - set(nc_keys)


def _read_nc_keys(nc, config_field):
    return nc.ncattrs() if 'attributes' in config_field else nc.variables.keys()


def _read_config_keys(config_field, file_name):
    file_type = utils.get_file_type(file_name)
    keys = CONFIG[config_field][file_type].split(',')
    return np.char.strip(keys)
