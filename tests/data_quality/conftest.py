import pytest
from tests import utils
import netCDF4
import numpy as np

CONFIG = utils.read_config('data_quality/data_quality_config.ini')


@pytest.fixture
def out_of_limits_values(pytestconfig):
    bad = {}
    nc = netCDF4.Dataset(pytestconfig.option.test_file)
    for var, limits in CONFIG.items('limits'):
        if var in nc.variables:
            limits = tuple(map(float, limits.split(',')))
            if np.min(nc.variables[var][:]) < limits[0]:
                bad[var] = 'min value out of limits'
            if np.max(nc.variables[var][:]) > limits[1]:
                bad[var] = 'max value out of limits'
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
