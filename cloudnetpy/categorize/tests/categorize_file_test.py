import pytest
import netCDF4
from tests.test import initialize_test_data

INSTRUMENT_LIST = ['categorize']
TEST_DATA_PATH = initialize_test_data(INSTRUMENT_LIST)


@pytest.fixture
def test_data():
    key_dict = {}
    for i, instrument in enumerate(INSTRUMENT_LIST):
        keys = list(netCDF4.Dataset(TEST_DATA_PATH[i]).variables.keys())
        key_dict[instrument] = keys
    return key_dict


def _error_msg(missing_keys, name):
    return f"Variable(s) {missing_keys} missing in {name} file!"


def test_categorize_file(test_data):
    must_keys = ['v', 'width', 'ldr', 'Z', 'v_sigma', 'Z_error', 'Z_sensitivity',
                 'Z_bias', 'latitude', 'longitude', 'altitude', 'time', 'height',
                 'radar_frequency', 'category_bits', 'insect_prob', 'is_rain',
                 'quality_bits', 'beta', 'model_time', 'temperature', 'q']
    missing_keys = must_keys - test_data['categorize']
    assert not missing_keys, _error_msg(missing_keys, 'categorize')
