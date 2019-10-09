"""Module for creating Cloudnet drizzle product.
"""
import os
from bisect import bisect_left
import numpy as np
import numpy.ma as ma
from scipy.special import gamma
import netCDF4
from cloudnetpy import utils, output
from cloudnetpy.categorize.categorize import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy.products.product_tools import ProductClassification


def generate_drizzle(categorize_file, output_file):
    """Generates Cloudnet drizzle product.

    This function calculates different drizzle properties from
    cloud radar and lidar measurements. The results are written in a netCDF file.

    Args:
        categorize_file (str): Categorize file name.
        output_file (str): Output file name.

    References:
        O’Connor, E.J., R.J. Hogan, and A.J. Illingworth, 2005:
        Retrieving Stratocumulus Drizzle Parameters Using Doppler Radar and Lidar.
        J. Appl. Meteor., 44, 14–27, https://doi.org/10.1175/JAM-2181.1

    """
    drizzle_data = DrizzleSource(categorize_file)
    drizzle_class = DrizzleClassification(categorize_file)
    width_ht = correct_spectral_width(categorize_file)
    drizzle_parameters = drizzle_solve(drizzle_data, drizzle_class, width_ht)
    derived_products = _calc_derived_products(drizzle_data, drizzle_parameters)
    errors = _calc_errors(drizzle_data, drizzle_parameters)
    results = {**drizzle_parameters, **derived_products, **errors}
    results = _screen_rain(results, drizzle_class)
    results['drizzle_retrieval_status'] = _get_retrieval_status(drizzle_class)
    _append_data(drizzle_data, results)
    output.update_attributes(drizzle_data.data, DRIZZLE_ATTRIBUTES)
    output.save_product_file('drizzle', drizzle_data, output_file)
    drizzle_data.close()


class DrizzleSource(DataSource):
    """Class holding the input data for drizzle calculations."""
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.mie = self._read_mie_lut()
        self.dheight = utils.mdiff(self.getvar('height'))
        self.z = self._convert_z_units()
        self.beta = self.getvar('beta')
        self.v = self.getvar('v')

    def _convert_z_units(self):
        """Converts reflectivity factor to SI units."""
        z = self.getvar('Z') - 180
        return utils.db2lin(z)

    def _read_mie_lut(self):
        """Reads mie scattering look-up table."""
        def _get_mie_file():
            module_path = os.path.dirname(os.path.abspath(__file__))
            return '/'.join((module_path, 'mie_lu_tables.nc'))

        def _get_wl_band():
            """Returns string corresponding the radar frequency."""
            radar_frequency = self.getvar('radar_frequency')
            wl_band = utils.get_wl_band(radar_frequency)
            return '35' if wl_band == 0 else '94'

        mie_file = _get_mie_file()
        nc = netCDF4.Dataset(mie_file)
        mie = nc.variables
        lut = {'Do': mie['lu_medianD'][:],
               'mu': mie['lu_u'][:],
               'S': mie['lu_k'][:],
               'lwf': mie['lu_LWF'][:],
               'termv': mie['lu_termv'][:]}
        band = _get_wl_band()
        lut.update({'width': mie[f"lu_width_{band}"][:],
                    'ray': mie[f"lu_mie_ray_{band}"][:],
                    'v': mie[f"lu_v_{band}"][:]})
        nc.close()
        return lut


class DrizzleClassification(ProductClassification):
    """Class storing the information about different drizzle types."""
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.is_v_sigma = self._find_v_sigma(categorize_file)
        self.warm_liquid = self._find_warm_liquid()
        self.drizzle = self._find_drizzle()
        self.would_be_drizzle = self._find_would_be_drizzle()
        self.cold_rain = self._find_cold_rain()

    @staticmethod
    def _find_v_sigma(cat_file):
        v_sigma = p_tools.read_nc_fields(cat_file, 'v_sigma')
        return np.isfinite(v_sigma)

    def _find_warm_liquid(self):
        return (self.category_bits['droplet']
                & ~self.category_bits['cold'])

    def _find_drizzle(self):
        return (~utils.transpose(self.is_rain)
                & self.category_bits['falling']
                & ~self.category_bits['droplet']
                & ~self.category_bits['cold']
                & ~self.category_bits['melting']
                & ~self.category_bits['insect']
                & self.quality_bits['radar']
                & self.quality_bits['lidar']
                & ~self.quality_bits['clutter']
                & ~self.quality_bits['molecular']
                & ~self.quality_bits['attenuated']
                & self.is_v_sigma)

    def _find_would_be_drizzle(self):
        return (~utils.transpose(self.is_rain)
                & self.warm_liquid
                & self.category_bits['falling']
                & ~self.category_bits['melting']
                & ~self.category_bits['insect']
                & self.quality_bits['radar']
                & ~self.quality_bits['clutter']
                & ~self.quality_bits['molecular'])

    def _find_cold_rain(self):
        return np.any(self.category_bits['melting'], axis=1)


def correct_spectral_width(cat_file):
    """Corrects spectral width.

    Removes the effect of turbulence and horizontal wind that cause
    spectral broadening of the Doppler velocity.

    Args:
        cat_file (str): Categorize file name.

    Returns:
        ndarray: Spectral width containing the correction for turbulence
            broadening.

    """
    def _calc_beam_divergence():
        beam_width = 0.5
        height = p_tools.read_nc_fields(cat_file, 'height')
        return height * np.deg2rad(beam_width)

    def _calc_v_sigma_factor():
        beam_divergence = _calc_beam_divergence()
        wind = calc_horizontal_wind(cat_file)
        actual_wind = (wind + beam_divergence) ** (2/3)
        scaled_wind = (30*wind + beam_divergence) ** (2/3)
        return actual_wind / (scaled_wind - actual_wind)

    width, v_sigma = p_tools.read_nc_fields(cat_file, ['width', 'v_sigma'])
    sigma_factor = _calc_v_sigma_factor()
    return width - sigma_factor * v_sigma


def calc_horizontal_wind(cat_file):
    """Calculates magnitude of horizontal wind.

    Args:
        cat_file: Categorize file name.

    Returns:
        ndarray: Horizontal wind (m s-1).

    """
    u_wind, v_wind = p_tools.interpolate_model(cat_file, ['uwind', 'vwind'])
    return utils.l2norm(u_wind, v_wind)


def drizzle_solve(data, drizzle_class, width_ht):
    """Estimates drizzle parameters.

    Args:
        data (DrizzleSource): Input data.
        drizzle_class (DrizzleClassification): Classification of the atmosphere
            for drizzle calculations.
        width_ht (ndarray): Corrected spectral width.

    """
    def _init_variables():
        shape = data.z.shape
        res = {'Do': np.zeros(shape), 'mu': np.zeros(shape),
               'S': np.zeros(shape), 'beta_corr': np.ones(shape)}
        return res, np.zeros(shape)

    def _calc_beta_z_ratio():
        return 2 / np.pi * data.beta / data.z

    def _find_lut_indices(*ind):
        ind_dia = bisect_left(data.mie['Do'], dia_init[ind], hi=n_dia-1)
        ind_width = bisect_left(width_lut[:, ind_dia], width_ht[ind], hi=n_widths-1)
        return ind_width, ind_dia

    def _update_result_tables(*ind):
        params['Do'][ind] = dia
        params['mu'][ind] = data.mie['mu'][lut_ind[0]]
        params['S'][ind] = data.mie['S'][lut_ind]

    def _is_converged(*ind):
        threshold = 1e-3
        return abs((dia - dia_init[ind]) / dia_init[ind]) < threshold

    params, dia_init = _init_variables()
    beta_z_ratio = _calc_beta_z_ratio()
    drizzle_ind = np.where(drizzle_class.drizzle == 1)
    dia_init[drizzle_ind] = calc_dia(beta_z_ratio[drizzle_ind], k=18.8)
    # Negation because width look-up table is descending order:
    width_lut = -data.mie['width'][:]
    n_widths, n_dia = width_lut.shape[0], len(data.mie['Do'])
    width_ht = -width_ht
    max_ite = 10
    for i, j in zip(*drizzle_ind):
        for _ in range(max_ite):
            lut_ind = _find_lut_indices(i, j)
            dia = calc_dia(beta_z_ratio[i, j] * params['beta_corr'][i, j],
                           data.mie['mu'][lut_ind[0]],
                           data.mie['ray'][lut_ind],
                           data.mie['S'][lut_ind])
            _update_result_tables(i, j)
            if _is_converged(i, j):
                break
            dia_init[i, j] = dia
        beta_factor = np.exp(2*params['S'][i, j]*data.beta[i, j]*data.dheight)
        params['beta_corr'][i, (j+1):] *= beta_factor
    return params


def calc_dia(beta_z_ratio, mu=0, ray=1, k=1):
    """ Drizzle diameter calculation.

    Args:
        beta_z_ratio (ndarray): Beta to z ratio, multiplied by (2 / pi).
        mu (ndarray, optional): Shape parameter for gamma calculations. Default is 0.
        ray (ndarray, optional): Mie to Rayleigh ratio for z. Default is 1.
        k (ndarray, optional): Alpha to beta ratio . Default is 1.

    References:
        https://journals.ametsoc.org/doi/pdf/10.1175/JAM-2181.1

    """
    const = ray * k * beta_z_ratio
    return (gamma(3 + mu) / gamma(7 + mu) * (3.67 + mu) ** 4 / const) ** (1/4)


def _calc_derived_products(data, parameters):
    """Calculates additional quantities from the drizzle properties."""
    def _calc_density():
        """Calculates drizzle number density (m-3)."""
        return data.z * 3.67 ** 6 / parameters['Do'] ** 6

    def _calc_lwc():
        """Calculates drizzle liquid water content (kg m-3)"""
        rho_water = 1000
        dia, mu, s = [parameters.get(key) for key in ('Do', 'mu', 'S')]
        gamma_ratio = gamma(4 + mu) / gamma(3 + mu) / (3.67 + mu)
        return rho_water / 3 * data.beta * s * dia * gamma_ratio

    def _calc_lwf(lwc_in):
        """Calculates drizzle liquid water flux."""
        flux = ma.copy(lwc_in)
        flux[ind_drizzle] *= data.mie['lwf'][ind_lut] * data.mie['termv'][ind_lut[1]]
        return flux

    def _calc_fall_velocity():
        """Calculates drizzle droplet fall velocity (m s-1)."""
        velocity = np.zeros_like(parameters['Do'])
        velocity[ind_drizzle] = -data.mie['v'][ind_lut]
        return velocity

    def _find_indices():
        drizzle_ind = np.where(parameters['Do'])
        ind_mu = np.searchsorted(data.mie['mu'], parameters['mu'][drizzle_ind])
        ind_dia = np.searchsorted(data.mie['Do'], parameters['Do'][drizzle_ind])
        n_widths, n_dia = len(data.mie['mu']), len(data.mie['Do'])
        ind_mu[ind_mu >= n_widths] = n_widths - 1
        ind_dia[ind_dia >= n_dia] = n_dia - 1
        return drizzle_ind, (ind_mu, ind_dia)

    def _calc_v_air(droplet_velocity):
        """Calculates vertical air velocity."""
        velocity = -np.copy(droplet_velocity)
        velocity[ind_drizzle] += data.v[ind_drizzle]
        return velocity

    ind_drizzle, ind_lut = _find_indices()
    density = _calc_density()
    lwc = _calc_lwc()
    lwf = _calc_lwf(lwc)
    v_drizzle = _calc_fall_velocity()
    v_air = _calc_v_air(v_drizzle)
    return {'drizzle_N': density, 'drizzle_lwc': lwc, 'drizzle_lwf': lwf,
            'v_drizzle': v_drizzle, 'v_air': v_air}


def _calc_errors(categorize, parameters):
    """Estimates errors in the retrieved drizzle products."""

    mu_error = 0.07
    mu_error_small = 0.25

    def _read_input_uncertainty(uncertainty_type):
        return tuple(db2lin(categorize.getvar(f'{key}_{uncertainty_type}'))
                     for key in ('Z', 'beta'))

    def _calc_parameter_errors():
        def _calc_dia_error():
            error = _calc_error(2/7, (1, 1), add_mu=True)
            error_small = _calc_error(1/4, (1, 1), add_mu_small=True)
            return _stack_errors(error, error_small)

        def _calc_lwc_error():
            error = _calc_error(1/7, (1, 6))
            error_small = _calc_error(1/4, (1, 3))
            return _stack_errors(error, error_small)

        def _calc_lwf_error():
            error = _calc_error(1/7, (3, 4), add_mu=True)
            error_small = _calc_error(1/2, (1, 1), add_mu_small=True)
            error_tiny = _calc_error(1/4, (3, 1), add_mu_small=True)
            return _stack_errors(error, error_small, error_tiny)

        def _calc_s_error():
            error = _calc_error(1/2, (1, 1))
            return _stack_errors(error)

        return {'Do_error': _calc_dia_error(),
                'drizzle_lwc_error': _calc_lwc_error(),
                'drizzle_lwf_error': _calc_lwf_error(),
                'S_error': _calc_s_error()}

    def _calc_error(scale, weights, add_mu=False, add_mu_small=False):
        error = utils.l2norm_weighted(error_input, scale, weights)
        if add_mu:
            error = utils.l2norm(error, mu_error)
        if add_mu_small:
            error = utils.l2norm(error, mu_error_small)
        return error

    def _stack_errors(error_in, error_small=None, error_tiny=None):
        def add_error_component(source, ind):
            error[ind] = source[ind]

        error = ma.zeros(error_in.shape)
        add_error_component(error_in, drizzle_indices['drizzle'])
        if error_small is not None:
            add_error_component(error_small, drizzle_indices['small'])
        if error_tiny is not None:
            add_error_component(error_tiny, drizzle_indices['tiny'])
        return error

    def _calc_parameter_biases():
        dia_bias = _calc_bias(2/7, (1, 1))
        lwc_bias = _calc_bias(1/7, (1, 6))
        lwf_bias = _calc_bias(1/7, (3, 4))
        return {'Do_bias': dia_bias,
                'drizzle_lwc_bias': lwc_bias,
                'drizzle_lwf_bias': lwf_bias}

    def _calc_bias(scale, weights):
        return utils.l2norm_weighted(bias_input, scale, weights)

    def _add_supplementary_errors():
        def _calc_n_error():
            z_error = error_input[0]
            dia_error = db2lin(results['Do_error'])
            n_error = utils.l2norm(z_error, 6*dia_error)
            return _stack_errors(n_error)

        def _calc_v_error():
            error = results['Do_error']
            error[drizzle_indices['tiny']] *= error[drizzle_indices['tiny']]
            return error

        results['drizzle_N_error'] = _calc_n_error()
        results['v_drizzle_error'] = _calc_v_error()
        results['mu_error'] = mu_error

    def _add_supplementary_biases():
        def _calc_n_bias():
            z_bias = bias_input[0]
            dia_bias = db2lin(results['Do_bias'])
            return utils.l2norm_weighted((z_bias, dia_bias), 1, (1, 6))

        results['drizzle_N_bias'] = _calc_n_bias()
        results['v_drizzle_bias'] = results['Do_bias']

    def _convert_to_db(data):
        """Converts linear error values to dB."""
        return {name: lin2db(value) for name, value in data.items()}

    drizzle_indices = _get_drizzle_indices(parameters['Do'])
    error_input = _read_input_uncertainty('error')
    bias_input = _read_input_uncertainty('bias')
    errors = _calc_parameter_errors()
    biases = _calc_parameter_biases()
    results = {**errors, **biases}
    _add_supplementary_errors()
    _add_supplementary_biases()
    return _convert_to_db(results)


def _get_drizzle_indices(diameter):
    return {'drizzle': diameter > 0,
            'small': np.logical_and(diameter <= 1e-4, diameter > 1e-5),
            'tiny': np.logical_and(diameter <= 1e-5, diameter > 0)}


def _screen_rain(results, classification):
    """Removes rainy profiles from drizzle variables.."""
    for key in results.keys():
        if not utils.isscalar(results[key]):
            results[key][classification.is_rain, :] = 0
    return results


def _get_retrieval_status(classification):
    def _find_retrieval_below_melting():
        cold_rain = utils.transpose(classification.cold_rain)
        below_melting = cold_rain * classification.drizzle
        status[below_melting == 1] = 2

    def _find_retrieval_in_warm_liquid():
        in_warm_liquid = (status == 0) * classification.warm_liquid
        status[in_warm_liquid == 1] = 4

    status = np.copy(classification.drizzle).astype(int)
    _find_retrieval_below_melting()
    status[classification.would_be_drizzle == 1] = 3
    _find_retrieval_in_warm_liquid()
    status[classification.is_rain == 1, :] = 5
    return status


def _append_data(drizzle_data, results):
    """Save retrieved fields to the drizzle_data object."""
    for key, value in results.items():
        value = ma.masked_where(value == 0, value)
        drizzle_data.append_data(value, key)


DRIZZLE_ATTRIBUTES = {
    'drizzle_N': MetaData(
        long_name='Drizzle number concentration',
        units='m-3',
        ancillary_variables='drizzle_N_error'
    ),
    'drizzle_N_error': MetaData(
        long_name='Random error in drizzle number concentration',
        units='dB'
    ),
    'drizzle_lwc': MetaData(
        long_name='Drizzle liquid water content',
        units='kg m-3',
        ancillary_variables='drizzle_lwc_error drizzle_lwc_bias'
    ),
    'drizzle_lwc_error': MetaData(
        long_name='Random error in drizzle liquid water content',
        units='dB',
    ),
    'drizzle_lwc_bias': MetaData(
        long_name='Possible bias in drizzle liquid water content',
        units='dB',
    ),
    'drizzle_lwf': MetaData(
        long_name='Drizzle liquid water flux',
        units='kg m-2 s-1',
        ancillary_variables='drizzle_lwf_error drizzle_lwf_bias'
    ),
    'drizzle_lwf_error': MetaData(
        long_name='Random error in drizzle liquid water flux',
        units='dB',
    ),
    'drizzle_lwf_bias': MetaData(
        long_name='Possible bias in drizzle liquid water flux',
        units='dB',
    ),
    'v_drizzle': MetaData(
        long_name='Drizzle droplet fall velocity',  # TODO: should it include 'terminal' ?
        units='m s-1',
        ancillary_variables='v_drizzle_error',
        positive='down'
    ),
    'v_drizzle_error': MetaData(
        long_name='Random error in drizzle droplet fall velocity',
        units='dB'
    ),
    'v_air': MetaData(
        long_name='Vertical air velocity',
        units='m s-1',
        ancillary_variables='v_air_error',
        positive='up',
    ),
    'v_air_error': MetaData(
        long_name='Random error in vertical air velocity',
        units='dB'
    ),
    'Do': MetaData(
        long_name='Drizzle median diameter',
        units='m',
        ancillary_variables='Do_error Do_bias'
    ),
    'Do_error': MetaData(
        long_name='Random error in drizzle median diameter',
        units='dB',
    ),
    'Do_bias': MetaData(
        long_name='Possible bias in drizzle median diameter',
        units='dB',
    ),
    'mu': MetaData(
        long_name='Drizzle droplet size distribution shape parameter',
        ancillary_variables='mu_error'
    ),
    'mu_error': MetaData(
        long_name='Random error in drizzle droplet size distribution shape parameter',
        units='dB',
    ),
    'S': MetaData(
        long_name='Lidar backscatter-to-extinction ratio',
        ancillary_variables='S_error'
    ),
    'S_error': MetaData(
        long_name='Random error in lidar backscatter-to-extinction ratio',
        units='dB'
    ),
    'beta_corr': MetaData(
        long_name='Lidar backscatter correction factor',
    ),
    'drizzle_retrieval_status': MetaData(
        long_name='Drizzle parameter retrieval status',
    )
}


# drizzle error linear / log conversion from the Matlab code:

COR = 10 / np.log(10)


def db2lin(x):
    return ma.exp(x / COR) - 1


def lin2db(x):
    return ma.log(x + 1) * COR
