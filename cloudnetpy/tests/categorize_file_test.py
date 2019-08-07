import pytest
import netCDF4
from tests.test import initialize_test_data

instrument_list = ['categorize']
test_data_path = initialize_test_data(instrument_list)


@pytest.fixture
def test_data():
    key_dict = {}
    for i, instrument in enumerate(instrument_list):
        keys = list(netCDF4.Dataset(test_data_path[i]).variables.keys())
        key_dict[instrument] = keys
    return key_dict


def test_categorize_file(test_data):
    must_keys = ['v', 'width', 'ldr', 'Z', 'v_sigma', 'Z_error', 'Z_sensitivity',
                 'Z_bias', 'latitude', 'longitude', 'altitude', 'time', 'height',
                 'radar_frequency', 'category_bits', 'insect_prob', 'is_rain',
                 'quality_bits', 'beta', 'model_time', 'temperature', 'q']
    compared_list = list(set(must_keys).intersection(test_data['categorize']))
    try:
        assert set(must_keys) == set(compared_list)
    except AssertionError:
        missing_variables = list(set(must_keys).difference(compared_list))
        raise Exception(f"Missing '{', '.join(missing_variables)}' variables in categorize file")
