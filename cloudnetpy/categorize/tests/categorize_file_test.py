import pytest
import netCDF4
from tests.test import initialize_test_data, missing_var_msg

INSTRUMENT_LIST = ['categorize']
TEST_DATA_PATH = initialize_test_data(INSTRUMENT_LIST)


@pytest.fixture
def test_data():
    key_dict = {}
    for path, instrument in zip(TEST_DATA_PATH, INSTRUMENT_LIST):
        key_dict[instrument] = set(netCDF4.Dataset(path).variables.keys())
    return key_dict


def test_categorize_file(test_data):
    must_keys = {'v', 'width', 'ldr', 'Z', 'v_sigma', 'Z_error', 'Z_sensitivity',
                 'Z_bias', 'latitude', 'longitude', 'altitude', 'time', 'height',
                 'radar_frequency', 'category_bits', 'insect_prob', 'is_rain',
                 'quality_bits', 'beta', 'model_time', 'temperature', 'q'}
    missing_keys = must_keys - test_data['categorize']
    assert not missing_keys, missing_var_msg(missing_keys, 'categorize')
