import numpy as np
import numpy.ma as ma
import cloudnetpy.utils as utils


def _get_drizzle_indices(diameter):
    return {'drizzle': diameter > 0,
            'small': np.logical_and(diameter <= 1e-4, diameter > 1e-5),
            'tiny': np.logical_and(diameter <= 1e-5, diameter > 0)}


def _read_input_uncertainty(categorize, uncertainty_type):
    return tuple(db2lin(categorize.getvar(f'{key}_{uncertainty_type}'))
                 for key in ('Z', 'beta'))


ERROR_INPUT = _read_input_uncertainty('error')
BIAS_INPUT = _read_input_uncertainty('bias')
MU_ERROR = 0.07
MU_ERROR_SMALL = 0.25


def get_drizzle_error(categorize, drizzle_parameters):
    parameters = drizzle_parameters.params
    drizzle_indices = _get_drizzle_indices(parameters['Do'])
    errors = _calc_errors(drizzle_indices)
    return errors


def _calc_errors(drizzle_indices):
    errors = _calc_parameter_errors(drizzle_indices)
    biases = _calc_parameter_biases()
    results = {**errors, **biases}
    _add_supplementary_errors(results, drizzle_indices)
    _add_supplementary_biases(results)
    return _convert_to_db(results)


def _calc_parameter_errors(drizzle_indices):
    def _calc_dia_error():
        error = _calc_error(2 / 7, (1, 1), add_mu=True)
        error_small = _calc_error(1 / 4, (1, 1), add_mu_small=True)
        return _stack_errors(error, drizzle_indices, error_small)

    def _calc_lwc_error():
        error = _calc_error(1 / 7, (1, 6))
        error_small = _calc_error(1 / 4, (1, 3))
        return _stack_errors(error, drizzle_indices, error_small)

    def _calc_lwf_error():
        error = _calc_error(1 / 7, (3, 4), add_mu=True)
        error_small = _calc_error(1 / 2, (1, 1), add_mu_small=True)
        error_tiny = _calc_error(1 / 4, (3, 1), add_mu_small=True)
        return _stack_errors(error, drizzle_indices, error_small, error_tiny)

    def _calc_s_error():
        error = _calc_error(1 / 2, (1, 1))
        return _stack_errors(error, drizzle_indices)

    return {'Do_error': _calc_dia_error(),
            'drizzle_lwc_error': _calc_lwc_error(),
            'drizzle_lwf_error': _calc_lwf_error(),
            'S_error': _calc_s_error()}


def _calc_parameter_biases():
    def _calc_bias(scale, weights):
        return utils.l2norm_weighted(BIAS_INPUT, scale, weights)

    dia_bias = _calc_bias(2/7, (1, 1))
    lwc_bias = _calc_bias(1/7, (1, 6))
    lwf_bias = _calc_bias(1/7, (3, 4))
    return {'Do_bias': dia_bias,
            'drizzle_lwc_bias': lwc_bias,
            'drizzle_lwf_bias': lwf_bias}


def _add_supplementary_errors(results, drizzle_indices):
    def _calc_n_error():
        z_error = ERROR_INPUT[0]
        dia_error = db2lin(results['Do_error'])
        n_error = utils.l2norm(z_error, 6 * dia_error)
        return _stack_errors(n_error, drizzle_indices)

    def _calc_v_error():
        error = results['Do_error']
        error[drizzle_indices['tiny']] *= error[drizzle_indices['tiny']]
        return error

    results['drizzle_N_error'] = _calc_n_error()
    results['v_drizzle_error'] = _calc_v_error()
    results['mu_error'] = MU_ERROR
    return results


def _add_supplementary_biases(results):
    def _calc_n_bias():
        z_bias = BIAS_INPUT[0]
        dia_bias = db2lin(results['Do_bias'])
        return utils.l2norm_weighted((z_bias, dia_bias), 1, (1, 6))

    results['drizzle_N_bias'] = _calc_n_bias()
    results['v_drizzle_bias'] = results['Do_bias']
    return results


def _convert_to_db(results):
    """Converts linear error values to dB."""
    return {name: lin2db(value) for name, value in results.items()}


def _calc_error(scale, weights, add_mu=False, add_mu_small=False):
    error = utils.l2norm_weighted(ERROR_INPUT, scale, weights)
    if add_mu:
        error = utils.l2norm(error, MU_ERROR)
    if add_mu_small:
        error = utils.l2norm(error, MU_ERROR_SMALL)
    return error


def _stack_errors(error_in, drizzle_indices, error_small=None, error_tiny=None):
    def _add_error_component(source, ind):
        error[ind] = source[ind]

    error = ma.zeros(error_in.shape)
    _add_error_component(error_in, drizzle_indices['drizzle'])
    if error_small is not None:
        _add_error_component(error_small, drizzle_indices['small'])
    if error_tiny is not None:
        _add_error_component(error_tiny, drizzle_indices['tiny'])
    return error


COR = 10 / np.log(10)


def db2lin(x):
    if ma.max(x) > 100:
        raise ValueError('Too large values in drizzle.db2lin()')
    return ma.exp(x / COR) - 1


def lin2db(x):
    if ma.min(x) < -0.9:
        raise ValueError('Too small values in drizzle.lin2db()')
    return ma.log(x + 1) * COR
