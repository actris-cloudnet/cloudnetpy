from tests.test import collect_variables, missing_var_msg

REQUIRED_KEYS = {
    'radar':
        {'Ze', 'v', 'width', 'ldr', 'SNR', 'radar_frequency', 'nfft',
         'prf', 'NyquistVelocity', 'nave', 'zrg', 'rg0', 'drg'},
    'ceilo':
        {'beta_raw', 'beta', 'beta_smooth', 'tilt_angle', 'height',
         'wavelength'}
}


def test_all():
    test_data = collect_variables(REQUIRED_KEYS.keys())
    for key in REQUIRED_KEYS:
        missing_keys = REQUIRED_KEYS[key] - test_data[key]
        assert not missing_keys, missing_var_msg(missing_keys, key)
