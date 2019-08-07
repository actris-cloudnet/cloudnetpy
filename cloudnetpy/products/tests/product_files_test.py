import pytest
import netCDF4
from tests.test import initialize_test_data

instrument_list = ['classification', 'iwc', 'lwc', 'drizzle']
test_data_path = initialize_test_data(instrument_list)


@pytest.fixture
def test_data():
    key_dict = {}
    for i, instrument in enumerate(instrument_list):
        keys = list(netCDF4.Dataset(test_data_path[i]).variables.keys())
        key_dict[instrument] = keys
    return key_dict


def test_classification_file(test_data):
    must_keys = ['target_classification', 'detection_status']
    compared_list = list(set(must_keys).intersection(test_data['classification']))
    try:
        assert set(must_keys) == set(compared_list)
    except AssertionError:
        missing_variables = list(set(must_keys).difference(compared_list))
        raise Exception(f"Missing '{', '.join(missing_variables)}' variables in classification file")


def test_iwc_file(test_data):
    must_keys = ['iwc', 'iwc_inc_rain', 'iwc_bias', 'iwc_error', 'iwc_sensitivity',
                 'iwc_retrieval_status']
    compared_list = list(set(must_keys).intersection(test_data['iwc']))
    try:
        assert set(must_keys) == set(compared_list)
    except AssertionError:
        missing_variables = list(set(must_keys).difference(compared_list))
        raise Exception(f"Missing '{', '.join(missing_variables)}' variables in iwc file")


def test_lwc_file(test_data):
    must_keys = ['lwc', 'lwc_error', 'lwc_retrieval_status', 'lwp', 'lwp_error']
    compared_list = list(set(must_keys).intersection(test_data['lwc']))
    try:
        assert set(must_keys) == set(compared_list)
    except AssertionError:
        missing_variables = list(set(must_keys).difference(compared_list))
        raise Exception(f"Missing '{', '.join(missing_variables)}' variables in lwc file")


def test_drizzle_file(test_data):
    must_keys = ['Do', 'mu', 'S', 'beta_corr', 'drizzle_N', 'drizzle_lwc', 'drizzle_lwf',
                 'v_drizzle', 'v_air', 'Do_error', 'drizzle_lwc_error',
                 'drizzle_lwf_error', 'S_error', 'Do_bias', 'drizzle_lwc_bias',
                 'drizzle_lwf_bias', 'drizzle_N_error', 'v_drizzle_error',
                 'drizzle_N_bias', 'v_drizzle_bias', 'drizzle_retrieval_status']
    compared_list = list(set(must_keys).intersection(test_data['drizzle']))
    try:
        assert set(must_keys) == set(compared_list)
    except AssertionError:
        missing_variables = list(set(must_keys).difference(compared_list))
        raise Exception(f"Missing '{', '.join(missing_variables)}' variables in drizzle file")
