from tests.test import collect_variables, missing_var_msg

REQUIRED_KEYS = {
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


def test_all():
    test_data = collect_variables(REQUIRED_KEYS.keys())
    for key in REQUIRED_KEYS:
        missing_keys = REQUIRED_KEYS[key] - test_data[key]
        assert not missing_keys, missing_var_msg(missing_keys, key)
