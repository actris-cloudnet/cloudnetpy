import pytest
from tests import utils
import netCDF4
import numpy as np

CONFIG = utils.read_config('data_quality/data_quality_config.ini')


@pytest.fixture
def data(pytestconfig):
    file = pytestconfig.option.test_file
    return Variable(file)


class Variable:
    def __init__(self, file):
        self.file_name = file
        self.bad_values = self._check_var_limits()

    def _check_var_limits(self):
        bad = {}
        nc = netCDF4.Dataset(self.file_name)
        nc_keys = nc.variables.keys()
        for var, limits in CONFIG.items('limits'):
            if var in nc_keys:
                limits = tuple(map(float, limits.split(',')))
                min_value = np.min(nc.variables[var][:])
                max_value = np.max(nc.variables[var][:])
                if min_value < limits[0] or max_value > limits[1]:
                    bad[var] = [min_value, max_value]
        nc.close()
        return bad
