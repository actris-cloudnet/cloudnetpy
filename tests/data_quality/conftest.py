import pytest
from tests import utils
import netCDF4
import numpy as np

CONFIG = utils.read_config('data_quality/data_quality_config.ini')


@pytest.fixture
def too_small_values(pytestconfig):
    return _test_limit(pytestconfig.option.test_file, 'lower')


@pytest.fixture
def too_large_values(pytestconfig):
    return _test_limit(pytestconfig.option.test_file, 'upper')


def _test_limit(file_name, limit):
    bad = []
    nc = netCDF4.Dataset(file_name)
    for var, limits in CONFIG.items('limits'):
        if var in nc.variables:
            limits = tuple(map(float, limits.split(',')))
            if limit == 'upper' and np.max(nc.variables[var][:]) > limits[1]:
                bad.append(var)
            elif limit == 'lower' and np.min(nc.variables[var][:]) < limits[0]:
                bad.append(var)
    nc.close()
    return bad


@pytest.fixture
def invalid_values(pytestconfig):
    bad = {}
    nc = netCDF4.Dataset(pytestconfig.option.test_file)
    for key in nc.variables:
        if np.isnan(nc.variables[key][:]).any():
            bad[key] = 'contains invalid values'
    nc.close()
    return bad
