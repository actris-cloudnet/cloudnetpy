"""Module for creating Cloudnet drizzle product.
"""
import os
from bisect import bisect_left
from typing import Union
import numpy as np
import numpy.ma as ma
from scipy.special import gamma
import netCDF4
from cloudnetpy import utils, output
from cloudnetpy.categorize import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy.products.product_tools import ProductClassification
from cloudnetpy.products.drizzle_error import get_drizzle_error


def generate_drizzle(categorize_file: str,
                     output_file: str,
                     keep_uuid: bool = False,
                     uuid: Union[str, None] = None) -> str:
    """Generates Cloudnet drizzle product.

    This function calculates different drizzle properties from
    cloud radar and lidar measurements. The results are written in a netCDF file.

    Args:
        categorize_file (str): Categorize file name.
        output_file (str): Output file name.
        keep_uuid (bool, optional): If True, keeps the UUID of the old file,
            if that exists. Default is False when new UUID is generated.
        uuid (str, optional): Set specific UUID for the file.

    Returns:
        str: UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_drizzle
        >>> generate_drizzle('categorize.nc', 'drizzle.nc')

    References:
        O’Connor, E.J., R.J. Hogan, and A.J. Illingworth, 2005:
        Retrieving Stratocumulus Drizzle Parameters Using Doppler Radar and Lidar.
        J. Appl. Meteor., 44, 14–27, https://doi.org/10.1175/JAM-2181.1

    """
    drizzle_source = DrizzleSource(categorize_file)
    drizzle_class = DrizzleClassification(categorize_file)
    spectral_width = SpectralWidth(categorize_file)
    drizzle_solver = DrizzleSolver(drizzle_source, drizzle_class,
                                   spectral_width)
    derived_products = DrizzleProducts(drizzle_source, drizzle_solver)
    errors = get_drizzle_error(drizzle_source, drizzle_solver)
    retrieval_status = RetrievalStatus(drizzle_class)
    results = {**drizzle_solver.params, **derived_products.derived_products,
               **errors}
    results = _screen_rain(results, drizzle_class)
    results['drizzle_retrieval_status'] = retrieval_status.retrieval_status
    _append_data(drizzle_source, results)
    date = drizzle_source.get_date()
    attributes = output.add_time_attribute(DRIZZLE_ATTRIBUTES, date)
    output.update_attributes(drizzle_source.data, attributes)
    uuid = output.save_product_file('drizzle', drizzle_source, output_file, keep_uuid, uuid)
    drizzle_source.close()
    return uuid


class DrizzleSource(DataSource):
    """Class holding the input data for drizzle calculations.

    Args:
        categorize_file (str): Categorize file name.

    Attributes:
        mie (dict): Mie look-up table data.
        dheight (float): Median difference of height array.
        z (ndarray): 2D radar echo (linear units).
        beta (ndarray): 2D lidar backscatter.
        v (ndarray): 2D doppler velocity.

    """
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
        mie_file = self._get_mie_file()
        nc = netCDF4.Dataset(mie_file)
        mie = nc.variables
        lut = {'Do': mie['lu_medianD'][:],
               'mu': mie['lu_u'][:],
               'S': mie['lu_k'][:],
               'lwf': mie['lu_LWF'][:],
               'termv': mie['lu_termv'][:]}
        band = self._get_wl_band()
        lut.update({'width': mie[f"lu_width_{band}"][:],
                    'ray': mie[f"lu_mie_ray_{band}"][:],
                    'v': mie[f"lu_v_{band}"][:]})
        nc.close()
        return lut

    @staticmethod
    def _get_mie_file():
        module_path = os.path.dirname(os.path.abspath(__file__))
        return '/'.join((module_path, 'mie_lu_tables.nc'))

    def _get_wl_band(self):
        """Returns string corresponding the radar frequency."""
        radar_frequency = self.getvar('radar_frequency')
        wl_band = utils.get_wl_band(radar_frequency)
        return '35' if wl_band == 0 else '94'


class DrizzleClassification(ProductClassification):
    """Class storing the information about different drizzle types,
    child of  :class:`ProductClassification`.

    Args:
        categorize_file (str): Categorize file name.

    Attributes:
        is_v_sigma (ndarray): 2D array denoting finite v_sigma.
        warm_liquid (ndarray): 2D array denoting warm liquid.
        drizzle (ndarray): 2D array denoting drizzle presence.
        would_be_drizzle (ndarray): 2D array denoting possible drizzle pixels.
        cold_rain (ndarray): 1D array denoting profiles with melting layer.

    """
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


class SpectralWidth:
    """Calculates corrected spectral width.

    Removes the effect of turbulence and horizontal wind that cause
    spectral broadening of the Doppler velocity.

    Args:
        categorize_file (str): Categorize file name.

    Attributes:
        categorize_file (str): Categorize file name.
        width_ht (ndarray): Spectral width containing the correction for
            turbulence broadening.

    """
    def __init__(self, categorize_file):
        self.cat_file = categorize_file
        self.width_ht = self._calculate_spectral_width()

    def _calculate_spectral_width(self):
        width, v_sigma = p_tools.read_nc_fields(self.cat_file, ['width', 'v_sigma'])
        sigma_factor = self._calc_v_sigma_factor()
        return width - sigma_factor * v_sigma

    def _calc_v_sigma_factor(self):
        beam_divergence = self._calc_beam_divergence()
        wind = self._calc_horizontal_wind()
        actual_wind = (wind + beam_divergence) ** (2/3)
        scaled_wind = (30*wind + beam_divergence) ** (2/3)
        return actual_wind / (scaled_wind - actual_wind)

    def _calc_beam_divergence(self):
        beam_width = 0.5
        height = p_tools.read_nc_fields(self.cat_file, 'height')
        return height * np.deg2rad(beam_width)

    def _calc_horizontal_wind(self):
        """Calculates magnitude of horizontal wind.

        Returns:
            ndarray: Horizontal wind (m s-1).

        """
        u_wind, v_wind = p_tools.interpolate_model(self.cat_file, ['uwind', 'vwind'])
        return utils.l2norm(u_wind, v_wind)


class DrizzleSolver:
    """Estimates drizzle parameters.

    Args:
        drizzle_source (DrizzleSource): The :class:`DrizzleSource` instance.
        drizzle_class (DrizzleClassification): The :class:`DrizzleClassification` instance.
        spectral_width (SpectralWidth): The :class:`SpectralWidth` instance.

    Attributes:
        params (dict): Dictionary of retrieved drizzle parameters 'Do', 'mu',
            'S', 'beta_corr'.

    """
    def __init__(self, drizzle_source, drizzle_class, spectral_width):
        self._data = drizzle_source
        self._drizzle_class = drizzle_class
        self._width_ht = spectral_width.width_ht
        self._width_lut = -self._data.mie['width'][:]
        self.params, self._dia_init = self._init_variables()
        self._beta_z_ratio = self._calc_beta_z_ratio()
        self._solve_drizzle(self._dia_init)

    def _init_variables(self):
        shape = self._data.z.shape
        res = {'Do': np.zeros(shape), 'mu': np.zeros(shape),
               'S': np.zeros(shape), 'beta_corr': np.ones(shape)}
        return res, np.zeros(shape)

    def _calc_beta_z_ratio(self):
        return 2 / np.pi * self._data.beta / self._data.z

    def _find_lut_indices(self, ind, dia_init, n_dia, n_widths):
        ind_dia = bisect_left(self._data.mie['Do'], dia_init[ind], hi=n_dia - 1)
        ind_width = bisect_left(self._width_lut[:, ind_dia], -self._width_ht[ind],
                                hi=n_widths-1)
        return ind_width, ind_dia

    def _update_result_tables(self, ind, dia, lut_ind):
        self.params['Do'][ind] = dia
        self.params['mu'][ind] = self._data.mie['mu'][lut_ind[0]]
        self.params['S'][ind] = self._data.mie['S'][lut_ind]

    @staticmethod
    def _is_converged(ind, dia, dia_init):
        threshold = 1e-3
        return abs((dia - dia_init[ind]) / dia_init[ind]) < threshold

    @staticmethod
    def _calc_dia(beta_z_ratio, mu=0, ray=1, k=1):
        """ Drizzle diameter calculation.

        Args:
            beta_z_ratio (ndarray): Beta to z ratio, multiplied by (2 / pi).
            mu (ndarray, optional): Shape parameter for gamma calculations. Default is 0.
            ray (ndarray, optional): Mie to Rayleigh ratio for z. Default is 1.
            k (ndarray, optional): Alpha to beta ratio . Default is 1.

        Returns:
            ndarray: Drizzle diameter.

        References:
            https://journals.ametsoc.org/doi/pdf/10.1175/JAM-2181.1

        """
        const = ray * k * beta_z_ratio
        return (gamma(3 + mu) / gamma(7 + mu) * (3.67 + mu) ** 4 / const) ** (1 / 4)

    def _solve_drizzle(self, dia_init):
        drizzle_ind = np.where(self._drizzle_class.drizzle == 1)
        dia_init[drizzle_ind] = self._calc_dia(self._beta_z_ratio[drizzle_ind], k=18.8)
        n_widths, n_dia = self._width_lut.shape[0], len(self._data.mie['Do'])
        max_ite = 10
        for ind in zip(*drizzle_ind):
            for _ in range(max_ite):
                lut_ind = self._find_lut_indices(ind, dia_init, n_dia, n_widths)
                dia = self._calc_dia(self._beta_z_ratio[ind] * self.params['beta_corr'][ind],
                                     self._data.mie['mu'][lut_ind[0]],
                                     self._data.mie['ray'][lut_ind],
                                     self._data.mie['S'][lut_ind])
                self. _update_result_tables(ind, dia, lut_ind)
                if self._is_converged(ind, dia, dia_init):
                    break
                self._dia_init[ind] = dia
            beta_factor = np.exp(2 * self.params['S'][ind] * self._data.beta[ind] * self._data.dheight)
            self.params['beta_corr'][ind[0], (ind[-1]+1):] *= beta_factor


class DrizzleProducts:
    """Calculates additional quantities from the drizzle properties.

    Args:
        drizzle_source (DrizzleSource): The :class:`DrizzleSource` instance.
        drizzle_solver (DrizzleSolver): The :class:`DrizzleSolver` instance.

    Attributes:
        derived_products (dict): Dictionary containing derived drizzle products:
            'drizzle_N', 'drizzle_lwc', 'drizzle_lwf', 'v_drizzle', 'v_air'.

    """
    def __init__(self, drizzle_source, drizzle_solver):
        self._data = drizzle_source
        self._params = drizzle_solver.params
        self._ind_drizzle, self._ind_lut = self._find_indices()
        self.derived_products = self._calc_derived_products()

    def _find_indices(self):
        drizzle_ind = np.where(self._params['Do'])
        ind_mu = np.searchsorted(self._data.mie['mu'], self._params['mu'][drizzle_ind])
        ind_dia = np.searchsorted(self._data.mie['Do'], self._params['Do'][drizzle_ind])
        n_widths, n_dia = len(self._data.mie['mu']), len(self._data.mie['Do'])
        ind_mu[ind_mu >= n_widths] = n_widths - 1
        ind_dia[ind_dia >= n_dia] = n_dia - 1
        return drizzle_ind, (ind_mu, ind_dia)

    def _calc_derived_products(self):
        density = self._calc_density()
        lwc = self._calc_lwc()
        lwf = self._calc_lwf(lwc)
        v_drizzle = self._calc_fall_velocity()
        v_air = self._calc_v_air(v_drizzle)
        return {'drizzle_N': density, 'drizzle_lwc': lwc, 'drizzle_lwf': lwf,
                'v_drizzle': v_drizzle, 'v_air': v_air}

    def _calc_density(self):
        """Calculates drizzle number density (m-3)."""
        a = self._data.z * 3.67 ** 6
        b = self._params['Do'] ** 6
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def _calc_lwc(self):
        """Calculates drizzle liquid water content (kg m-3)"""
        rho_water = 1000
        dia, mu, s = [self._params.get(key) for key in ('Do', 'mu', 'S')]
        gamma_ratio = gamma(4 + mu) / gamma(3 + mu) / (3.67 + mu)
        return rho_water / 3 * self._data.beta * s * dia * gamma_ratio

    def _calc_lwf(self, lwc_in):
        """Calculates drizzle liquid water flux."""
        flux = ma.copy(lwc_in)
        flux[self._ind_drizzle] *= (self._data.mie['lwf'][self._ind_lut]
                                    * self._data.mie['termv'][self._ind_lut[1]])
        return flux

    def _calc_fall_velocity(self):
        """Calculates drizzle droplet fall velocity (m s-1)."""
        velocity = np.zeros_like(self._params['Do'])
        velocity[self._ind_drizzle] = -self._data.mie['v'][self._ind_lut]
        return velocity

    def _calc_v_air(self, droplet_velocity):
        """Calculates vertical air velocity."""
        velocity = -np.copy(droplet_velocity)
        velocity[self._ind_drizzle] += self._data.v[self._ind_drizzle]
        return velocity


class RetrievalStatus:
    """Estimates the status of drizzle retrievals.

    Args:
        drizzle_class (DrizzleClassification): The :class:`DrizzleClassification` instance.

    Attributes:
        drizzle_class (DrizzleClassification): The :class:`DrizzleClassification` instance.
        retrieval_status (ndarray): 2D array containing drizzle retrieval status
            information.
    """
    def __init__(self, drizzle_class):
        self.drizzle_class = drizzle_class
        self.retrieval_status = None
        self._get_retrieval_status()

    def _get_retrieval_status(self):
        self.retrieval_status = np.copy(self.drizzle_class.drizzle).astype(int)
        self._find_retrieval_below_melting()
        self.retrieval_status[self.drizzle_class.would_be_drizzle == 1] = 3
        self._find_retrieval_in_warm_liquid()
        self.retrieval_status[self.drizzle_class.is_rain == 1, :] = 5

    def _find_retrieval_below_melting(self):
        cold_rain = utils.transpose(self.drizzle_class.cold_rain)
        below_melting = cold_rain * self.drizzle_class.drizzle
        self.retrieval_status[below_melting == 1] = 2

    def _find_retrieval_in_warm_liquid(self):
        in_warm_liquid = (self.retrieval_status == 0) * self.drizzle_class.warm_liquid
        self.retrieval_status[in_warm_liquid == 1] = 4


def _screen_rain(results, classification):
    """Removes rainy profiles from drizzle variables.."""
    for key in results.keys():
        if not utils.isscalar(results[key]):
            results[key][classification.is_rain, :] = 0
    return results


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
