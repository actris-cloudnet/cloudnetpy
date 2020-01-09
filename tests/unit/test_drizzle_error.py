import numpy as np
import numpy.testing as testing
import pytest
from cloudnetpy.products import drizzle_error


def test_get_drizzle_indices():
    dia = np.array([-1, 2 * 1e-5, 1, 1e-6])
    d = drizzle_error._get_drizzle_indices(dia)
    correct = {'drizzle': [False, True, True, True],
               'small': [False, True, False, False],
               'tiny': [False, False, False, True]}
    for key in d.keys():
        testing.assert_array_equal(d[key], correct[key])


def test_read_input_uncertainty():
    assert True


def test_calc_parameter_errors():
    assert True


def test_calc_parameter_biases():
    assert True


def test_add_supplementary_errors():
    assert True


def test_calc_v_error():
    assert True


def test_add_supplementary_biases():
    assert True


def test_convert_to_db():
    assert True


def test_calc_error():
    assert True


def test_stack_errors():
    assert True


def test_add_error_component():
    assert True


@pytest.mark.parametrize("x, result", [
    (-1000, -1),
    (-100, -0.99999),
    (-10, -0.9),
    (-1, np.exp(-1 / 10 * np.log(10)) - 1),
])
def test_db2lin(x, result):
    testing.assert_array_almost_equal(drizzle_error.db2lin(x), result, decimal=5)


def test_db2lin_raise():
    with pytest.raises(ValueError):
        drizzle_error.db2lin(150)


@pytest.mark.parametrize("x, result", [
    (1e6, 60),
    (1e5, 50),
    (1e4, 40),
])
def test_lin2db(x, result):
    testing.assert_array_almost_equal(drizzle_error.lin2db(x), result, decimal=3)


def test_lin2db_raise():
    with pytest.raises(ValueError):
        drizzle_error.lin2db(-1)

