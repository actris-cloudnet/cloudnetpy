from tests.test import read_variable_names, missing_var_msg

REQUIRED_KEYS = {
    'radar':
        {'Ze', 'v', 'width', 'ldr', 'SNR', 'radar_frequency', 'nfft',
         'prf', 'NyquistVelocity', 'nave', 'zrg', 'rg0', 'drg'},
    'ceilo':
        {'beta_raw', 'beta', 'beta_smooth', 'tilt_angle', 'height',
         'wavelength'}
    }


def test_all():
    for key in REQUIRED_KEYS:
        keys_in_test_file = read_variable_names(key)
        missing_keys = REQUIRED_KEYS[key] - keys_in_test_file
        assert not missing_keys, missing_var_msg(missing_keys, key)
