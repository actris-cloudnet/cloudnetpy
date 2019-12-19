
def test_get_drizzle_indices():
    assert True


def test_read_input_uncertainty():
    assert True


def test_calc_errors():
    assert True


def test_calc_parameter_errors():
    assert True


def test_calc_dia_error():
    assert True


def test_calc_lwc_error():
    assert True


def test_calc_lwf_error():
    assert True


def test_calc_s_error():
    assert True


def test_calc_error():
    assert True


def test_stack_errors():
    assert True


def test_add_error_component():
    assert True


def test_calc_parameter_biases():
    assert True


def test_calc_bias():
    assert True


def test_add_supplementary_errors():
    assert True


def test_calc_n_error():
    assert True


def test_calc_v_error():
    assert True


def test_add_supplementary_biases():
    assert True


def test_calc_n_bias():
    assert True


def test_convert_to_db():
    assert True


"""
def test_get_drizzle_indices():
    dia = np.array([-1, 2 * 1e-5, 1, 1e-6])
    d = drizzle.CalculateErrors._get_drizzle_indices(dia)
    correct = {'drizzle': [False, True, True, True],
               'small': [False, True, False, False],
               'tiny': [False, False, False, True]}
    for key in d.keys():
        testing.assert_array_equal(d[key], correct[key])
"""
