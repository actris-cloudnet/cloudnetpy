import numpy as np
from numpy import ma

from cloudnetpy import utils
from cloudnetpy.products.drizzle_tools import DrizzleSolver, DrizzleSource

MU_ERROR = 0.07
MU_ERROR_SMALL = 0.25


def get_drizzle_error(categorize: DrizzleSource, drizzle_parameters: DrizzleSolver) -> dict:
    """Estimates error and bias for drizzle classification.

    Args:
        categorize: The :class:`DrizzleSource` instance.
        drizzle_parameters: The :class:`DrizzleSolver` instance.

    Returns:
        dict: Dictionary containing information of estimated error and bias for drizzle

    """
    parameters = drizzle_parameters.params
    drizzle_indices = _get_drizzle_indices(parameters["Do"])
    error_input = _read_input_uncertainty(categorize, "error")
    if utils.isscalar(error_input[0]) is True:  # Constant Z error
        z_error, bias_error = error_input
        z_error = np.full(categorize.z.shape, z_error)
        error_input = z_error, bias_error
    bias_input = _read_input_uncertainty(categorize, "bias")
    errors = _calc_errors(drizzle_indices, error_input, bias_input)
    return errors


def _get_drizzle_indices(diameter: np.ndarray) -> dict:
    return {
        "drizzle": diameter > 0,
        "small": np.logical_and(diameter <= 1e-4, diameter > 1e-5),
        "tiny": np.logical_and(diameter <= 1e-5, diameter > 0),
    }


def _read_input_uncertainty(categorize: DrizzleSource, uncertainty_type: str) -> tuple:
    return tuple(db2lin(categorize.getvar(f"{key}_{uncertainty_type}")) for key in ("Z", "beta"))


def _calc_errors(drizzle_indices: dict, error_input: tuple, bias_input: tuple) -> dict:
    errors = _calc_parameter_errors(drizzle_indices, error_input)
    biases = _calc_parameter_biases(bias_input)
    results = {**errors, **biases}
    _add_supplementary_errors(results, drizzle_indices, error_input)
    _add_supplementary_biases(results, bias_input)
    return _convert_to_db(results)


def _calc_parameter_errors(drizzle_indices: dict, error_input: tuple) -> dict:
    def _calc_dia_error() -> ma.MaskedArray:
        error = _calc_error(2 / 7, (1, 1), error_input, add_mu=True)
        error_small = _calc_error(1 / 4, (1, 1), error_input, add_mu_small=True)
        return _stack_errors(error, drizzle_indices, error_small)

    def _calc_lwc_error() -> ma.MaskedArray:
        error = _calc_error(1 / 7, (1, 6), error_input)
        error_small = _calc_error(1 / 4, (1, 3), error_input)
        return _stack_errors(error, drizzle_indices, error_small)

    def _calc_lwf_error() -> ma.MaskedArray:
        error = _calc_error(1 / 7, (3, 4), error_input, add_mu=True)
        error_small = _calc_error(1 / 2, (1, 1), error_input, add_mu_small=True)
        error_tiny = _calc_error(1 / 4, (3, 1), error_input, add_mu_small=True)
        return _stack_errors(error, drizzle_indices, error_small, error_tiny)

    def _calc_s_error() -> ma.MaskedArray:
        error = _calc_error(1 / 2, (1, 1), error_input)
        return _stack_errors(error, drizzle_indices)

    return {
        "Do_error": _calc_dia_error(),
        "drizzle_lwc_error": _calc_lwc_error(),
        "drizzle_lwf_error": _calc_lwf_error(),
        "S_error": _calc_s_error(),
    }


def _calc_parameter_biases(bias_input: tuple) -> dict:
    def _calc_bias(scale: float, weights: tuple) -> ma.MaskedArray:
        return utils.l2norm_weighted(bias_input, scale, weights)

    dia_bias = _calc_bias(2 / 7, (1, 1))
    lwc_bias = _calc_bias(1 / 7, (1, 6))
    lwf_bias = _calc_bias(1 / 7, (3, 4))
    return {"Do_bias": dia_bias, "drizzle_lwc_bias": lwc_bias, "drizzle_lwf_bias": lwf_bias}


def _add_supplementary_errors(results: dict, drizzle_indices: dict, error_input: tuple) -> dict:
    def _calc_n_error() -> ma.MaskedArray:
        z_error = error_input[0]
        dia_error = db2lin(results["Do_error"])
        n_error = utils.l2norm(z_error, 6 * dia_error)
        return _stack_errors(n_error, drizzle_indices)

    def _calc_v_error() -> ma.MaskedArray:
        error = results["Do_error"]
        error[drizzle_indices["tiny"]] *= error[drizzle_indices["tiny"]]
        return error

    results["drizzle_N_error"] = _calc_n_error()
    results["v_drizzle_error"] = _calc_v_error()
    results["mu_error"] = MU_ERROR
    return results


def _add_supplementary_biases(results: dict, bias_input: tuple) -> dict:
    def _calc_n_bias() -> ma.MaskedArray:
        z_bias = bias_input[0]
        dia_bias = db2lin(results["Do_bias"])
        return utils.l2norm_weighted((z_bias, dia_bias), 1, (1, 6))

    results["drizzle_N_bias"] = _calc_n_bias()
    results["v_drizzle_bias"] = results["Do_bias"]
    return results


def _convert_to_db(results: dict) -> dict:
    """Converts linear error values to dB."""
    return {name: lin2db(value) for name, value in results.items()}


def _calc_error(
    scale: float,
    weights: tuple,
    error_input: tuple,
    add_mu: bool = False,
    add_mu_small: bool = False,
) -> ma.MaskedArray:
    error = utils.l2norm_weighted(error_input, scale, weights)
    if add_mu is True:
        error = utils.l2norm(error, MU_ERROR)
    if add_mu_small is True:
        error = utils.l2norm(error, MU_ERROR_SMALL)
    return error


def _stack_errors(
    error_in: np.ndarray, drizzle_indices: dict, error_small=None, error_tiny=None
) -> ma.MaskedArray:
    def _add_error_component(source: np.ndarray, ind: tuple):
        error[ind] = source[ind]

    error = ma.zeros(error_in.shape)
    _add_error_component(error_in, drizzle_indices["drizzle"])
    if error_small is not None:
        _add_error_component(error_small, drizzle_indices["small"])
    if error_tiny is not None:
        _add_error_component(error_tiny, drizzle_indices["tiny"])
    return error


COR = 10 / np.log(10)


def db2lin(x_in: np.ndarray) -> ma.MaskedArray:
    x = ma.copy(x_in)
    threshold = 100
    x[x > threshold] = threshold
    return ma.exp(x / COR) - 1


def lin2db(x_in) -> ma.MaskedArray:
    x = ma.copy(x_in)
    threshold = -0.9
    x[x < threshold] = threshold
    return ma.log(x + 1) * COR
