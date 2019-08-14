from tests.test import read_variable_names, missing_var_msg


def test_categorize_file():
    required_keys = {
        'v', 'width', 'ldr', 'Z', 'v_sigma', 'Z_error', 'Z_sensitivity',
        'Z_bias', 'latitude', 'longitude', 'altitude', 'time', 'height',
        'radar_frequency', 'category_bits', 'insect_prob', 'is_rain',
        'quality_bits', 'beta', 'model_time', 'temperature', 'q'}
    identifier = 'categorize'
    keys_in_test_file = read_variable_names(identifier)
    missing_keys = required_keys - keys_in_test_file
    assert not missing_keys, missing_var_msg(missing_keys, identifier)
