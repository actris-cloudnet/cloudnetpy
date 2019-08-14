"""Tests for CloudnetPy categorize file."""
from tests.test import read_variable_names, missing_var_msg


def test_required_variables():
    required_variables = {
        'v', 'width', 'ldr', 'Z', 'v_sigma', 'Z_error', 'Z_sensitivity',
        'Z_bias', 'latitude', 'longitude', 'altitude', 'time', 'height',
        'radar_frequency', 'category_bits', 'insect_prob', 'is_rain',
        'quality_bits', 'beta', 'model_time', 'temperature', 'q'}
    identifier = 'categorize'
    test_file_variables = read_variable_names(identifier)
    missing_variables = required_variables - test_file_variables
    assert not missing_variables, missing_var_msg(missing_variables, identifier)


def test_values():
    pass
