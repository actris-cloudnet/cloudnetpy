import numpy as np
import pytest
from numpy import testing

from cloudnetpy.products import drizzle_error as de

DRIZZLE_INDICES = {
    "drizzle": np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    "small": np.array([[1, 0, 0], [0, 0, 1]], dtype=bool),
    "tiny": np.array([[0, 0, 1], [0, 1, 0]], dtype=bool),
}
ERROR_INPUT = (np.array([[0.01, 0.34, 0.5], [0.2, 0.3, 0.56]]), 0.14)
BIAS_INPUT = (0.01, 0.57)
MU = np.array([[0.0, 2.0, 5.0], [10.0, -0.5, -1.0]])


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


@pytest.mark.parametrize(
    "key",
    ["Do_error", "drizzle_lwc_error", "drizzle_lwf_error", "S_error"],
)
def test_calc_parameter_errors(key):
    x = de._calc_parameter_errors(DRIZZLE_INDICES, ERROR_INPUT)
    assert key in x.keys()


@pytest.mark.parametrize("key", ["Do_bias", "drizzle_lwc_bias", "drizzle_lwf_bias"])
def test_calc_parameter_biases(key):
    x = de._calc_parameter_biases(BIAS_INPUT)
    assert key in x.keys()


@pytest.fixture()
def results():
    errors = de._calc_parameter_errors(DRIZZLE_INDICES, ERROR_INPUT)
    biases = de._calc_parameter_biases(BIAS_INPUT)
    return {**errors, **biases}


@pytest.mark.parametrize("key", ["drizzle_N_error", "v_drizzle_error", "mu_error"])
def test_add_supplementary_errors(results, key):
    x = de._add_supplementary_errors(results, DRIZZLE_INDICES, ERROR_INPUT, MU)
    assert key in x.keys()


def test_calc_v_error(results):
    results["Do_error"] = np.array([[2, 2, 2], [2, 2, 2]])
    x = de._add_supplementary_errors(results, DRIZZLE_INDICES, ERROR_INPUT, MU)
    testing.assert_almost_equal(x["v_drizzle_error"][DRIZZLE_INDICES["tiny"]], 4)


def test_calc_n_error(results):
    indices = {key: np.ones((2, 3), dtype=bool) for key in DRIZZLE_INDICES}
    dia_error = np.full((2, 3), 0.3)
    results["Do_error"] = dia_error.copy()
    x = de._add_supplementary_errors(results, indices, ERROR_INPUT, MU)
    mu_error = de._calc_n_mu_error(MU)
    expected = np.sqrt(ERROR_INPUT[0] ** 2 + (6 * dia_error) ** 2 + mu_error**2)
    testing.assert_allclose(x["drizzle_N_error"], expected)


def test_calc_n_mu_error_matches_numerical_derivative():
    from scipy.special import gammaln

    def log_density(mu):
        return gammaln(1 + mu) - gammaln(7 + mu) + 6 * np.log(3.67 + mu)

    mu = np.array([-0.9, -0.5, 0.0, 2.0, 5.0, 10.0])
    step = 1e-6
    derivative = (log_density(mu + step) - log_density(mu - step)) / (2 * step)
    expected = np.abs(derivative) * de.MU_STEP
    testing.assert_allclose(de._calc_n_mu_error(mu), expected, rtol=1e-5)


def test_calc_n_mu_error_is_zero_for_undefined_density():
    result = de._calc_n_mu_error(np.array([-1.0, -1.5]))
    testing.assert_array_equal(result, 0)


def test_calc_n_bias(results):
    x = de._add_supplementary_biases(results, BIAS_INPUT)
    expected = np.sqrt(BIAS_INPUT[0] ** 2 + (6 * results["Do_bias"]) ** 2)
    testing.assert_allclose(x["drizzle_N_bias"], expected)


@pytest.mark.parametrize("key", ["drizzle_N_bias", "v_drizzle_bias"])
def test_add_supplementary_biases(results, key):
    x = de._add_supplementary_biases(results, BIAS_INPUT)
    assert key in x.keys()


def test_calc_error():
    from cloudnetpy.utils import l2norm_weighted

    expected = l2norm_weighted(ERROR_INPUT, 1, (1,))
    testing.assert_almost_equal(de._calc_error(1, (1,), ERROR_INPUT), expected)


def test_stack_errors():
    DRIZZLE_INDICES["drizzle"] = np.array([[0, 1, 1], [1, 1, 0]], dtype=bool)
    expected = np.ma.array(
        ERROR_INPUT[0], mask=[[True, False, False], [False, False, True]]
    )
    x = de._stack_errors(ERROR_INPUT[0], DRIZZLE_INDICES)
    testing.assert_array_almost_equal(x, expected)


@pytest.mark.parametrize(
    "x, result",
    [
        (-1000, -1),
        (-100, -0.99999),
        (-10, -0.9),
        (1000, np.exp(100 / 10 * np.log(10)) - 1),
    ],
)
def test_db2lin(x, result):
    testing.assert_array_almost_equal(de.db2lin(x), result, decimal=3)


@pytest.mark.parametrize("x, result", [(1e6, 60), (1e5, 50), (1e4, 40), (-100.0, -10)])
def test_lin2db(x, result):
    testing.assert_array_almost_equal(de.lin2db(x), result, decimal=3)
