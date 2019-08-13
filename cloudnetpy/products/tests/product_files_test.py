from tests.test import collect_variables, missing_var_msg

INSTRUMENT_LIST = ['classification', 'iwc', 'lwc', 'drizzle']
TEST_DATA = collect_variables(INSTRUMENT_LIST)


def test_classification_file():
    must_keys = {'target_classification', 'detection_status'}
    missing_keys = must_keys - TEST_DATA['classification']
    assert not missing_keys, missing_var_msg(missing_keys, 'classification')


def test_iwc_file():
    must_keys = {'iwc', 'iwc_inc_rain', 'iwc_bias', 'iwc_error', 'iwc_sensitivity',
                 'iwc_retrieval_status'}
    missing_keys = must_keys - TEST_DATA['iwc']
    assert not missing_keys, missing_var_msg(missing_keys, 'iwc')


def test_lwc_file():
    must_keys = {'lwc', 'lwc_error', 'lwc_retrieval_status', 'lwp', 'lwp_error'}
    missing_keys = must_keys - TEST_DATA['lwc']
    assert not missing_keys, missing_var_msg(missing_keys, 'lwc')


def test_drizzle_file():
    must_keys = {'Do', 'mu', 'S', 'beta_corr', 'drizzle_N', 'drizzle_lwc', 'drizzle_lwf',
                 'v_drizzle', 'v_air', 'Do_error', 'drizzle_lwc_error',
                 'drizzle_lwf_error', 'S_error', 'Do_bias', 'drizzle_lwc_bias',
                 'drizzle_lwf_bias', 'drizzle_N_error', 'v_drizzle_error',
                 'drizzle_N_bias', 'v_drizzle_bias', 'drizzle_retrieval_status'}
    missing_keys = must_keys - TEST_DATA['drizzle']
    assert not missing_keys, missing_var_msg(missing_keys, 'drizzle')
