from tests.test import collect_variables, missing_var_msg

MUST_BE_KEYS = {
    'radar':
        {'Ze', 'v', 'width', 'ldr', 'SNR', 'radar_frequency', 'nfft',
         'prf', 'NyquistVelocity', 'nave', 'zrg', 'rg0', 'drg'},
    'ceilo':
        {'beta_raw', 'beta', 'beta_smooth', 'tilt_angle', 'height',
         'wavelength'}
}


def test_all():
    test_data = collect_variables(MUST_BE_KEYS.keys())
    for key in MUST_BE_KEYS:
        missing_keys = MUST_BE_KEYS[key] - test_data[key]
        assert not missing_keys, missing_var_msg(missing_keys, key)
