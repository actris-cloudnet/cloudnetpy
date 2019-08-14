"""Tests for raw radar/lidar files, and model / hatpro files."""
from tests.test import missing_var_msg, read_variable_names, read_attribute_names, missing_attr_msg

REQUIRED_KEYS = {
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
}


def test_required_keys():
    for key in REQUIRED_KEYS:
        keys_in_test_file = read_variable_names(key)
        missing_keys = REQUIRED_KEYS[key] - keys_in_test_file
        assert not missing_keys, missing_var_msg(missing_keys, key)


def test_required_attributes():
    for key in REQUIRED_ATTRIBUTES:
        attributes_in_test_file = read_attribute_names(key)
        missing_attributes = REQUIRED_ATTRIBUTES[key] - attributes_in_test_file
        assert not missing_attributes, missing_attr_msg(missing_attributes, key)
