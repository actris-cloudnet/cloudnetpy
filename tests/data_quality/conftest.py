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
    bad = {}
    nc = netCDF4.Dataset(file_name)
    for var, limits in CONFIG.items('limits'):
        if var in nc.variables:
            limits = tuple(map(float, limits.split(',')))
            if limit == 'upper':
                value = np.max(nc.variables[var][:])
                if value > limits[1]:
                    bad[var] = (np.max(nc.variables[var][:]), limits[1])
            elif limit == 'lower':
                value = np.min(nc.variables[var][:])
                if value < limits[0]:
                    bad[var] = (np.min(nc.variables[var][:]), limits[0])
    nc.close()
    return bad


@pytest.fixture
def invalid_values(pytestconfig):
    bad = []
    nc = netCDF4.Dataset(pytestconfig.option.test_file)
    for key in nc.variables:
        var = nc.variables[key][:]
        if np.isnan(var).any() or np.isnan(var).any():
            bad.append(key)
    nc.close()
    return bad
