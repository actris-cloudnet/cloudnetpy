import pytest
import netCDF4
from tests.test import initialize_test_data

INSTRUMENT_LIST = ['classification', 'iwc', 'lwc', 'drizzle']
TEST_DATA_PATH = initialize_test_data(INSTRUMENT_LIST)


@pytest.fixture
def test_data():
    key_dict = {}
    for i, instrument in enumerate(INSTRUMENT_LIST):
        keys = list(netCDF4.Dataset(TEST_DATA_PATH[i]).variables.keys())
        key_dict[instrument] = keys
    return key_dict


def _error_msg(missing_keys, name):
    return f"Variable(s) {missing_keys} missing in {name} file!"


def test_classification_file(test_data):
    must_keys = ['target_classification', 'detection_status']
    missing_keys = must_keys - test_data['classification']
    assert not missing_keys, _error_msg(missing_keys, 'classification')


def test_iwc_file(test_data):
    must_keys = ['iwc', 'iwc_inc_rain', 'iwc_bias', 'iwc_error', 'iwc_sensitivity',
                 'iwc_retrieval_status']
    missing_keys = must_keys - test_data['iwc']
    assert not missing_keys, _error_msg(missing_keys, 'iwc')


def test_lwc_file(test_data):
    must_keys = ['lwc', 'lwc_error', 'lwc_retrieval_status', 'lwp', 'lwp_error']
    missing_keys = must_keys - test_data['lwp']
    assert not missing_keys, _error_msg(missing_keys, 'lwp')


def test_drizzle_file(test_data):
    must_keys = ['Do', 'mu', 'S', 'beta_corr', 'drizzle_N', 'drizzle_lwc', 'drizzle_lwf',
                 'v_drizzle', 'v_air', 'Do_error', 'drizzle_lwc_error',
                 'drizzle_lwf_error', 'S_error', 'Do_bias', 'drizzle_lwc_bias',
                 'drizzle_lwf_bias', 'drizzle_N_error', 'v_drizzle_error',
                 'drizzle_N_bias', 'v_drizzle_bias', 'drizzle_retrieval_status']
    missing_keys = must_keys - test_data['drizzle']
    assert not missing_keys, _error_msg(missing_keys, 'drizzle')
