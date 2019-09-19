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
    return CollectMetadata(request.param, file_name)


class CollectMetadata:
    def __init__(self, keys_in, file):
        self.config = meta_config
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