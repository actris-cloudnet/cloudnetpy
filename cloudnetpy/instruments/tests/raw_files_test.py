from tests.test import missing_var_msg, read_variable_names

REQUIRED_KEYS = {
    'mira_raw':
        {'nfft', 'prf', 'nave', 'zrg', 'rg0', 'drg', 'NyquistVelocity',
         'time', 'range', 'Zg', 'VELg', 'RMSg', 'LDRg', 'SNRg'},
    'chm15k_raw':
        {'latitude', 'longitude', 'azimuth', 'zenith', 'time', 'range',
         'layer', 'altitude', 'wavelength', 'laser_pulses', 'beta_raw'},
    'hatpro':
        {'time_reference', 'minimum', 'maximum', 'time', 'rain_flag',
         'elevation_angle', 'azimuth_angle', 'retrieval', 'LWP_data'},
    'ecmwf':
        {'latitude', 'longitude', 'horizontal_resolution', 'time',
         'forecast_time', 'level', 'pressure', 'uwind', 'vwind', 'omega',
         'temperature', 'q', 'rh'}
    }


def test_all():
    for key in REQUIRED_KEYS:
        keys_in_test_file = read_variable_names(key)
        missing_keys = REQUIRED_KEYS[key] - keys_in_test_file
        assert not missing_keys, missing_var_msg(missing_keys, key)
