from tests.test import collect_variables, missing_var_msg

INSTRUMENT_LIST = ['categorize']
TEST_DATA = collect_variables(INSTRUMENT_LIST)


def test_categorize_file():
    must_keys = {'v', 'width', 'ldr', 'Z', 'v_sigma', 'Z_error', 'Z_sensitivity',
                 'Z_bias', 'latitude', 'longitude', 'altitude', 'time', 'height',
                 'radar_frequency', 'category_bits', 'insect_prob', 'is_rain',
                 'quality_bits', 'beta', 'model_time', 'temperature', 'q'}
    missing_keys = must_keys - TEST_DATA['categorize']
    assert not missing_keys, missing_var_msg(missing_keys, 'categorize')
