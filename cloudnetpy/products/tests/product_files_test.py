"""Tests for CloudnetPy product files."""
from tests.test import read_variable_names, missing_key_msg, read_attribute_names

REQUIRED_COMMON_ATTRIBUTES = {
    'Conventions', 'cloudnetpy_version', 'file_uuid', 'title', 'source', 'year',
    'month', 'day', 'location', 'history'
}

REQUIRED_COMMON_VARIABLES = {
    'latitude', 'longitude', 'altitude', 'time', 'height'
}

REQUIRED_VARIABLES = {
    'classification':
        {'target_classification', 'detection_status'},
    'iwc':
        {'iwc', 'iwc_inc_rain', 'iwc_bias', 'iwc_error', 'iwc_sensitivity',
         'iwc_retrieval_status'},
    'lwc':
        {'lwc', 'lwc_error', 'lwc_retrieval_status', 'lwp', 'lwp_error'},
    'drizzle':
        {'Do', 'mu', 'S', 'beta_corr', 'drizzle_N', 'drizzle_lwc', 'drizzle_lwf',
         'v_drizzle', 'v_air', 'Do_error', 'drizzle_lwc_error',
         'drizzle_lwf_error', 'S_error', 'Do_bias', 'drizzle_lwc_bias',
         'drizzle_lwf_bias', 'drizzle_N_error', 'v_drizzle_error',
         'drizzle_N_bias', 'v_drizzle_bias', 'drizzle_retrieval_status'}
}


def test_required_keys():

    def _is_missing(required, observed, is_attr=False):
        missing = required - observed
        assert not missing, missing_key_msg(missing, key, is_attr)

    for key in REQUIRED_VARIABLES:
        test_file_attributes = read_attribute_names(key)
        test_file_variables = read_variable_names(key)
        _is_missing(REQUIRED_COMMON_ATTRIBUTES, test_file_attributes, True)
        _is_missing(REQUIRED_COMMON_VARIABLES, test_file_variables)
        _is_missing(REQUIRED_VARIABLES[key], test_file_variables)
