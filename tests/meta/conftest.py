import pytest
import netCDF4
from tests.utils import read_config, find_missing_keys, check_var_limits

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
        self._nc = netCDF4.Dataset(file)
        self._keys = self._nc.ncattrs()
        self.bad_units = self._check_attr_units()
        self.bad_values = self._check_attr_limits()
        self._nc.close()

    def _check_attr_units(self):
        bad = {}
        for attr, reference in CONFIG.items('attributes_units'):
            if attr in self._keys:
                value = self._nc.getncattr(attr)
                if reference != value:
                    bad[attr] = value
        return bad

    def _check_attr_limits(self):
        bad = {}
        for attr, limits in CONFIG.items('attributes_limits'):
            if attr in self._keys:
                limits = tuple(map(float, limits.split(',')))
                value = int(self._nc.getncattr(attr))
                if value < limits[0] or value > limits[1]:
                    bad[attr] = value
        return bad


class Variable:
    def __init__(self, file):
        self.file_name = file
        self.bad_values = check_var_limits(CONFIG, 'variables_limits', file)
        self.bad_units = self._check_units()

    def _check_units(self):
        bad = {}
        nc = netCDF4.Dataset(self.file_name)
        keys = nc.variables.keys()
        for var, reference in CONFIG.items('variables_units'):
            if var in keys:
                value = nc.variables[var].units
                if reference != value:
                    bad[var] = value
        nc.close()
        return bad
