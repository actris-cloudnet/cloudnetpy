"""Tests for raw radar/lidar files, and model / hatpro files."""
from tests.test import missing_var_msg, read_variable_names, read_attribute_names, missing_attr_msg

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
}


def test_required_variables():
    for key in REQUIRED_VARIABLES:
        test_file_variables = read_variable_names(key)
        missing_variables = REQUIRED_VARIABLES[key] - test_file_variables
        assert not missing_variables, missing_var_msg(missing_variables, key)


def test_required_attributes():
    for key in REQUIRED_ATTRIBUTES:
        test_file_attributes = read_attribute_names(key)
        missing_attributes = REQUIRED_ATTRIBUTES[key] - test_file_attributes
        assert not missing_attributes, missing_attr_msg(missing_attributes, key)
