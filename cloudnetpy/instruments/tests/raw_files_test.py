"""Tests for raw radar/lidar files, and model / hatpro files."""
from tests.test import missing_key_msg, read_variable_names, read_attribute_names

REQUIRED_VARIABLES = {
    'mira_raw':
        {'prf', 'NyquistVelocity', 'time', 'range', 'Zg', 'VELg', 'RMSg',
         'LDRg', 'SNRg'},
    'chm15k_raw':
        {'time', 'range', 'beta_raw', 'zenith', 'wavelength'},
    'hatpro':
        {'time', 'LWP_data', 'elevation_angle'},
    'ecmwf':
        {'temperature', 'pressure', 'rh', 'gas_atten', 'specific_gas_atten',
         'specific_saturated_gas_atten', 'specific_liquid_atten', 'q', 'uwind',
         'vwind', 'height', 'time'}
}

REQUIRED_ATTRIBUTES = {
    'mira_raw':
        {'Latitude', 'Longitude'},
    'chm15k_raw':
        {'year', 'month', 'day'},
    'hatpro':
        {'station_altitude'},
    'ecmwf':
        {'history'}
}


def test_required_keys():

    def _is_missing(required, observed, is_attr=False):
        missing = required - observed
        assert not missing, missing_key_msg(missing, key, is_attr)

    for key in REQUIRED_VARIABLES:
        test_file_attributes = read_attribute_names(key)
        test_file_variables = read_variable_names(key)
        _is_missing(REQUIRED_ATTRIBUTES[key], test_file_attributes, True)
        _is_missing(REQUIRED_VARIABLES[key], test_file_variables)
