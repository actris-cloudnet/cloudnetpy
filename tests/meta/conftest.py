import pytest
import netCDF4
from tests import utils

CONFIG = utils.read_config('meta/metadata_config.ini')


@pytest.fixture
def missing_variables(pytestconfig):
    file_name = pytestconfig.option.test_file
    return utils.find_missing_keys(CONFIG, 'required_variables', file_name)


@pytest.fixture
def missing_global_attributes(pytestconfig):
    file_name = pytestconfig.option.test_file
    return utils.find_missing_keys(CONFIG, 'required_attributes', file_name)


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
        self.bad_units = _check_units('attributes_units', file)

    def _check_values(self):
        bad = {}
        nc = netCDF4.Dataset(self.file_name)
        keys = nc.ncattrs()
        for attr, limits in CONFIG.items('attributes_limits'):
            if attr in keys:
                limits = tuple(map(float, limits.split(',')))
                value = int(nc.getncattr(attr))
                if not limits[0] <= value <= limits[1]:
                    bad[attr] = value
        nc.close()
        return bad


class Variable:
    def __init__(self, file):
        self.file_name = file
        self.bad_values = utils.check_var_limits(CONFIG, 'variables_limits', file)
        self.bad_units = _check_units('variables_units', file)


def _check_units(config_field, file_name):
    bad = {}
    nc = netCDF4.Dataset(file_name)
    keys = nc.ncattrs() if 'attributes' in config_field else nc.variables.keys()
    for var, reference in CONFIG.items(config_field):
        if var in keys:
            value = nc.variables[var].units
            if reference != value:
                bad[var] = value
    nc.close()
    return bad
