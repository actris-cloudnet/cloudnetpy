from tests.test import collect_variables, missing_var_msg

INSTRUMENT_LIST = ['mira_raw', 'chm15k_raw', 'hatpro', 'ecmwf']
TEST_DATA = collect_variables(INSTRUMENT_LIST)


def test_mira_raw_file():
    must_keys = {'nfft', 'prf', 'nave', 'zrg', 'rg0', 'drg', 'NyquistVelocity',
                 'time', 'range', 'Zg', 'VELg', 'RMSg', 'LDRg', 'SNRg'}
    missing_keys = must_keys - TEST_DATA['mira_raw']
    assert not missing_keys, missing_var_msg(missing_keys, 'raw mira')


def test_ceilo_raw_file():
    must_keys = {'latitude', 'longitude', 'azimuth', 'zenith', 'time', 'range',
                 'layer', 'altitude', 'wavelength', 'laser_pulses', 'beta_raw'}
    missing_keys = must_keys - TEST_DATA['chm15k_raw']
    assert not missing_keys, missing_var_msg(missing_keys, 'raw lidar')


def test_model_file():
    must_keys = {'latitude', 'longitude', 'horizontal_resolution', 'time', 'forecast_time',
                 'level', 'pressure', 'uwind', 'vwind', 'omega', 'temperature',
                 'q', 'rh'}
    missing_keys = must_keys - TEST_DATA['ecmwf']
    assert not missing_keys, missing_var_msg(missing_keys, 'ecmwf model')


def test_mwr_file():
    must_keys = {'time_reference', 'minimum', 'maximum', 'time', 'rain_flag',
                 'elevation_angle', 'azimuth_angle', 'retrieval', 'LWP_data'}
    missing_keys = must_keys - TEST_DATA['hatpro']
    assert not missing_keys, missing_var_msg(missing_keys, 'hatpro')
