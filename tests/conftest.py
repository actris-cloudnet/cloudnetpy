import pytest
import netCDF4


@pytest.fixture
def variable_names(pytestconfig):
    file = pytestconfig.option.test_file
    nc = netCDF4.Dataset(file)
    keys = set(nc.variables.keys())
    nc.close()
    return keys


@pytest.fixture
def global_attribute_names(pytestconfig):
    file = pytestconfig.option.test_file
    nc = netCDF4.Dataset(file)
    keys = set(nc.ncattrs())
    nc.close()
    return keys


@pytest.fixture
def global_attribute(request, pytestconfig):
    file = pytestconfig.option.test_file
    return GlobalAttribute(file, request.param)


@pytest.fixture
def variable(request, pytestconfig):
    file = pytestconfig.option.test_file
    return Variable(file, request.param)


class GlobalAttribute:
    def __init__(self, file_name, attr_name):
        self.name = attr_name
        self.is_attribute = False
        self.value = None
        self._read_attr(file_name)

    def _read_attr(self, file):
        nc = netCDF4.Dataset(file)
        self.is_attribute = hasattr(nc, self.name)
        self.value = getattr(nc, self.name, None)
        nc.close()


class Variable:
    def __init__(self, file_name, var_name):
        self.name = var_name
        self.is_variable = False
        self.value = None
        self.units = None
        self._read_var(file_name)

    def _read_var(self, file):
        nc = netCDF4.Dataset(file)
        if self.name in nc.variables:
            self.is_variable = True
            self.value = nc.variables[self.name][:]
            self.units = getattr(nc.variables[self.name], 'units', None)
        nc.close()


def pytest_addoption(parser):
    parser.addoption('--test_file', action='store', help='Input file name')
