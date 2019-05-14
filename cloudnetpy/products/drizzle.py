"""Module for creating Cloudnet drizzle product.
"""
import os
import numpy as np
import numpy.ma as ma
import netCDF4
from cloudnetpy import utils
from cloudnetpy.categorize import DataSource
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy.products.product_tools import ProductClassification
from cloudnetpy.plotting import plot_2d
from scipy.special import gamma


def generate_drizzle(categorize_file, output_file):
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
        self.z = self._get_z()
        self.beta = self.getvar('beta')

    def _get_z(self):
        """Converts reflectivity factor to linear space."""
        z = self.getvar('Z') - 180  # what's this 180 ?
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
        self.warm_liquid = self._find_warm_liquid()
        self.drizzle = self._find_drizzle()
        self.would_be_drizzle = self._find_would_be_drizzle()
        self.cold_rain = self._find_cold_rain()

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
                & ~self.quality_bits['attenuated'])

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
    """Calculates magnitude of horizontal wind."""
    u_wind, v_wind = p_tools.interpolate_model(cat_file, ['uwind', 'vwind'])
    return utils.l2norm(u_wind, v_wind)


def calc_s(p, q, const, mu, beta, k):
    """ Help function for gamma-calculations """

    a = gamma(mu + p) / gamma(mu + (p - q))
    b = 1 / ((mu + 3.67) ** q)
    c = const * beta * k
    return a * b * c


def calc_dia(z, beta, mu, ray, k):
    """ Drizzle diameter calculation

    Args:
        z (ndarray): Radar reflectivity factor in linear units.
        beta (ndarray): Ceilometer backscatter (m-1 sr-1)
        mu (ndarray): Shape parameter for gamma calculations.
        ray (ndarray): Mie to Rayleigh ratio for z.
        k (ndarray): Unknown parameter.

    """
    p, q = 7, 3
    const = 2 / np.pi * ray / z
    s = calc_s(p, q, const, mu, beta, k)
    return (1 / s) ** (1 / (p - q))


def drizzle_solve(data, drizzle_class, width_ht):
    """Estimates drizzle parameters.

    Args:
        data (DrizzleSource): Input data.
        drizzle_class (DrizzleClassification): Classification of the atmosphere
            for drizzle calculations.
        width_ht (ndarray): 2D array of turbulence.

    """
    shape = data.z.shape
    diameter, diameter_old, tab_mu = utils.init(3, shape)
    beta_corr = np.ones(shape)
    tab_mie_ray = np.ones(shape)
    init_k = 18.8
    tab_k = np.full(shape, init_k)
    drizzle_ind = np.where(drizzle_class.drizzle)
    diameter_old[drizzle_ind] = calc_dia(data.z[drizzle_ind],
                                         data.beta[drizzle_ind]*init_k,
                                         0, 1, 1)
    threshold = 1e-3
    max_ite = 10
    for i, j in zip(*drizzle_ind):

        old_dia = diameter_old[i, j]
        converged = False
        n_ite = 1
        while not converged and n_ite < max_ite:

            dia_ind = utils.nearest(data.mie['diameter'], old_dia)
            mu_ind = utils.nearest(data.mie['width'][:, dia_ind], width_ht[i, j])

            tab_mu[i, j] = data.mie['u'][mu_ind]
            tab_k[i, j] = data.mie['k'][mu_ind, dia_ind]
            tab_mie_ray[i, j] = data.mie['ray'][mu_ind, dia_ind]

            loop_dia = calc_dia(data.z[i, j],
                                data.beta[i, j],
                                tab_mu[i, j],
                                tab_mie_ray[i, j],
                                tab_k[i, j])

            if abs(loop_dia - old_dia) < threshold:
                diameter[i, j] = loop_dia
                converged = True
            else:
                old_dia = loop_dia
                n_ite += 1

        beta_factor = np.exp(2*tab_k[i, j]*data.beta[i, j]*data.dheight)
        beta_corr[i, (j+1):] *= beta_factor

    return diameter, tab_mu, tab_k, tab_mie_ray, beta_corr

