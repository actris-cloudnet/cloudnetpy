from tests.test import collect_variables, missing_var_msg

INSTRUMENT_LIST = ['radar', 'ceilo']
TEST_DATA = collect_variables(INSTRUMENT_LIST)


def test_radar_file():
    must_keys = {'Ze', 'v', 'width', 'ldr', 'SNR', 'radar_frequency', 'nfft',
                 'prf', 'NyquistVelocity', 'nave', 'zrg', 'rg0', 'drg'}
    missing_keys = must_keys - TEST_DATA['radar']
    assert not missing_keys, missing_var_msg(missing_keys, 'calibrated mira')


def test_ceilo_file():
    must_keys = {'beta_raw', 'beta', 'beta_smooth', 'tilt_angle', 'height',
                 'wavelength'}
    missing_keys = must_keys - TEST_DATA['ceilo']
    assert not missing_keys, missing_var_msg(missing_keys, 'calibrated ceilo')
