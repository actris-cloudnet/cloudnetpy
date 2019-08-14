"""Tests for calibrated CloudnetPy files."""
from tests.test import read_variable_names, missing_var_msg

REQUIRED_VARIABLES = {
    'radar':
        {'Ze', 'v', 'width', 'ldr', 'SNR', 'radar_frequency', 'nfft',
         'prf', 'NyquistVelocity', 'nave', 'zrg', 'rg0', 'drg'},
    'ceilo':
        {'beta_raw', 'beta', 'beta_smooth', 'tilt_angle', 'height',
         'wavelength'}
    }


def test_required_variables():
    for key in REQUIRED_VARIABLES:
        test_file_variables = read_variable_names(key)
        missing_variables = REQUIRED_VARIABLES[key] - test_file_variables
        assert not missing_variables, missing_var_msg(missing_variables, key)
