import pytest
import netCDF4
from tests.quality_control import read_config, get_test_file_name
from tests.metadata_control import read_meta_config

data_config = read_config()
file_name = get_test_file_name(data_config)


@pytest.fixture
def data(request):
    return InputData(request.param, file_name)


class InputData:
    def __init__(self, keys_in, file):
        self.config = data_config
        self.vrs = netCDF4.Dataset(file).variables
        self.variables = [self.vrs[key][:] for key in keys_in]
        self.keys = keys_in
        self.min = self._read_limits('min')
        self.max = self._read_limits('max')

    def _read_limits(self, limit):
        def _read_min_max():
            for key in self.keys:
                yield float(data_config['min_max'][key.lower()].split(',')[ind])
        ind = 0 if limit == 'min' else 1
        return _read_min_max()


meta_config = read_meta_config()
meta_file_name = get_test_file_name(meta_config)


@pytest.fixture
def metadata(request):
    return CollectMetadata(request.param, meta_file_name)


class CollectMetadata:
    def __init__(self, keys_in, file):
        self.config = meta_config
        self.vrs = netCDF4.Dataset(file).variables.keys()
        self.attr = netCDF4.Dataset(file).ncattrs()
        self.variables = [self.vrs[key][:] for key in keys_in]
        self.keys = keys_in

    def _read_required_attributes(self, data_name):
        req_attributes = self.config['required_attributes'][data_name]
        # datan lukemiseen configista pitää ehkä tehdä jotain extraa
        return req_attributes - self.attr

    def _read_required_variables(self, data_name):
        req_variables = self.config['required_variables'][data_name]
        return req_variables - self.vrs

    def _read_min_max_values(self, data_name):
        quantity_limits = self.config['units']

        return ""

    def _read_variable_units(self):
        print("")
        return ""
