"""Tests for CloudnetPy categorize file."""
import pytest

REQUIRED_VARIABLES = {
    'v', 'width', 'ldr', 'Z', 'v_sigma', 'Z_error', 'Z_sensitivity',
    'Z_bias', 'latitude', 'longitude', 'altitude', 'time', 'height',
    'radar_frequency', 'category_bits', 'insect_prob', 'is_rain',
    'quality_bits', 'beta', 'model_time', 'temperature', 'q'
}


@pytest.mark.categorize
def test_required_variables(variable_names):
    missing_variables = REQUIRED_VARIABLES - variable_names
    assert not missing_variables
