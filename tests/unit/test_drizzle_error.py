import numpy as np
import numpy.testing as testing
import pytest

from cloudnetpy.products import drizzle_error as de

DRIZZLE_PARAMETERS = {"Do": np.array([[0.0001, 0.01, 0.000001], [0.001, 0.000001, 0.0001]])}
DRIZZLE_INDICES = {
    "drizzle": np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    "small": np.array([[1, 0, 0], [0, 0, 1]], dtype=bool),
    "tiny": np.array([[0, 0, 1], [0, 1, 0]], dtype=bool),
}
ERROR_INPUT = (np.array([[0.01, 0.34, 0.5], [0.2, 0.3, 0.56]]), 0.14)
BIAS_INPUT = (0.01, 0.57)


@pytest.mark.parametrize(
    "key, value",
    [
        ("drizzle", [False, True, True, True]),
        ("small", [False, True, False, False]),
        ("tiny", [False, False, False, True]),
    ],
)
def test_get_drizzle_indices(key, value):
    dia = np.array([-1, 2 * 1e-5, 1, 1e-6])
    d = de._get_drizzle_indices(dia)
    testing.assert_array_equal(d[key], value)


@pytest.mark.parametrize("key", ["Do_error", "drizzle_lwc_error", "drizzle_lwf_error", "S_error"])
def test_calc_parameter_errors(key):
    x = de._calc_parameter_errors(DRIZZLE_INDICES, ERROR_INPUT)
    assert key in x.keys()


@pytest.mark.parametrize("key", ["Do_bias", "drizzle_lwc_bias", "drizzle_lwf_bias"])
def test_calc_parameter_biases(key):
    x = de._calc_parameter_biases(BIAS_INPUT)
    assert key in x.keys()


@pytest.fixture
def results():
    errors = de._calc_parameter_errors(DRIZZLE_INDICES, ERROR_INPUT)
    biases = de._calc_parameter_biases(BIAS_INPUT)
    return {**errors, **biases}


@pytest.mark.parametrize("key", ["drizzle_N_error", "v_drizzle_error", "mu_error"])
def test_add_supplementary_errors(results, key):
    x = de._add_supplementary_errors(results, DRIZZLE_INDICES, ERROR_INPUT)
    assert key in x.keys()


def test_calc_v_error(results):
    results["Do_error"] = np.array([[2, 2, 2], [2, 2, 2]])
    x = de._add_supplementary_errors(results, DRIZZLE_INDICES, ERROR_INPUT)
    testing.assert_almost_equal(x["v_drizzle_error"][DRIZZLE_INDICES["tiny"]], 4)


@pytest.mark.parametrize("key", ["drizzle_N_bias", "v_drizzle_bias"])
def test_add_supplementary_biases(results, key):
    x = de._add_supplementary_biases(results, BIAS_INPUT)
    assert key in x.keys()


def test_calc_error():
    from cloudnetpy.utils import l2norm_weighted

    expected = l2norm_weighted(ERROR_INPUT, 1, 1)
    testing.assert_almost_equal(de._calc_error(1, 1, ERROR_INPUT), expected)


def test_stack_errors():
    DRIZZLE_INDICES["drizzle"] = np.array([[0, 1, 1], [1, 1, 0]], dtype=bool)
    expected = np.ma.array(ERROR_INPUT[0], mask=[[1, 0, 0], [0, 0, 1]])
    x = de._stack_errors(ERROR_INPUT[0], DRIZZLE_INDICES)
    testing.assert_array_almost_equal(x, expected)


@pytest.mark.parametrize(
    "x, result",
    [(-1000, -1), (-100, -0.99999), (-10, -0.9), (1000, np.exp(100 / 10 * np.log(10)) - 1)],
)
def test_db2lin(x, result):
    testing.assert_array_almost_equal(de.db2lin(x), result, decimal=3)


@pytest.mark.parametrize("x, result", [(1e6, 60), (1e5, 50), (1e4, 40), (-100.0, -10)])
def test_lin2db(x, result):
    testing.assert_array_almost_equal(de.lin2db(x), result, decimal=3)
