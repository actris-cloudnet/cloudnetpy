"""Module for creating Cloudnet drizzle product.
"""
import os
from bisect import bisect_left
import numpy as np
from scipy.special import gamma
import netCDF4
from cloudnetpy import utils
from cloudnetpy.categorize import DataSource
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy.products.product_tools import ProductClassification


def generate_drizzle(categorize_file, output_file):
    """Generates Cloudnet drizzle product.

    Args:
        categorize_file (str): Categorize file name.
        output_file (str): Output file name.

    """
    drizzle_data = DrizzleSource(categorize_file)
    drizzle_class = DrizzleClassification(categorize_file)
    width_ht = correct_spectral_width(categorize_file)
    results = drizzle_solve(drizzle_data, drizzle_class, width_ht)


class DrizzleSource(DataSource):
    """Class holding the input data for drizzle calculations."""
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.mie = self._read_mie_lut()
        self.dheight = utils.mdiff(self.getvar('height'))
        self.z = self._convert_z_units()
        self.beta = self.getvar('beta')

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
        mie = netCDF4.Dataset(mie_file).variables
        lut = {'diameter': mie['lu_medianD'][:],
               'u': mie['lu_u'][:],
               'k': mie['lu_k'][:]}
        band = _get_wl_band()
        lut.update({'width': mie[f"lu_width_{band}"],
                    'ray': mie[f"lu_mie_ray_{band}"]})
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
        actual_wind = (wind + beam_divergence) ** (2 / 3)
        scaled_wind = (30 * wind + beam_divergence) ** (2 / 3)
        return actual_wind / (scaled_wind-actual_wind)

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


def drizzle_solve(data, drizzle_class, width_ht):
    """Estimates drizzle parameters.

    Args:
        data (DrizzleSource): Input data.
        drizzle_class (DrizzleClassification): Classification of the atmosphere
            for drizzle calculations.
        width_ht (ndarray): Corrected spectral width.

    """
    def _calc_beta_z_ratio():
        return 2 / np.pi * data.beta / data.z

    def _find_lut_indices(*ind):
        ind_dia = np.searchsorted(data.mie['diameter'], dia_init[ind])
        ind_mu = bisect_left(width_lut[:, ind_dia], width_ht[ind], hi=n_widths - 1)
        return ind_mu, ind_dia

    def _update_result_tables(*ind):
        params['dia'][ind] = loop_dia
        params['mu'][ind] = data.mie['u'][lut_ind[0]]
        params['k'][ind] = data.mie['k'][lut_ind]

    shape = data.z.shape
    params = dict.fromkeys(('dia', 'mu', 'k'), np.zeros(shape))
    dia_init, beta_corr = np.zeros(shape), np.ones(shape)
    beta_z_ratio = _calc_beta_z_ratio()
    drizzle_ind = np.where(drizzle_class.drizzle == 1)
    dia_init[drizzle_ind] = calc_dia(beta_z_ratio[drizzle_ind], k=18.8)
    # We have use negation because width should be ascending order
    width_lut = -data.mie['width'][:]
    n_widths = width_lut.shape[0]
    width_ht = -width_ht
    threshold, max_ite = 1e-9, 10
    for i, j in zip(*drizzle_ind):
        for _ in range(max_ite):
            lut_ind = _find_lut_indices(i, j)
            loop_dia = calc_dia(beta_z_ratio[i, j],
                                data.mie['u'][lut_ind[0]],
                                data.mie['ray'][lut_ind],
                                data.mie['k'][lut_ind])
            _update_result_tables(i, j)
            if abs(loop_dia - dia_init[i, j]) < threshold:
                break
            dia_init[i, j] = loop_dia
        beta_factor = np.exp(2*params['k'][i, j]*data.beta[i, j]*data.dheight)
        beta_corr[i, (j+1):] *= beta_factor
    return params
