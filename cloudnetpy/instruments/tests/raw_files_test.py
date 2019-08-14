from tests.test import collect_variables, missing_var_msg

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
    test_data = collect_variables(REQUIRED_KEYS.keys())
    for key in REQUIRED_KEYS:
        missing_keys = REQUIRED_KEYS[key] - test_data[key]
        assert not missing_keys, missing_var_msg(missing_keys, key)
