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
    width_ht = estimate_turb_sigma(categorize_file)


class DrizzleSource(DataSource):
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.mie = self._read_mie_data()
        print(self.mie)

    def _read_mie_data(self):
        """Reads mie scattering look-up table."""
        mie_file = self._get_mie_file()
        mie = netCDF4.Dataset(mie_file).variables
        lut = {'diameter': mie['lu_medianD'][:],
               'u': mie['lu_u'][:],
               'k': mie['lu_k'][:]}
        band = self._get_wl_band()
        lut.update({'width': mie[f"lu_width_{band}"],
                    'ray': mie[f"lu_mie_ray_{band}"]})
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


def estimate_turb_sigma(cat_file):
    """Not really sure what this function returns."""
    beamwidth = 0.5
    width, v_sigma, height = p_tools.read_nc_fields(cat_file, ['width', 'v_sigma', 'height'])
    uwind, vwind = p_tools.interpolate_model(cat_file, ['uwind', 'vwind'])
    wind = utils.l2norm(uwind, vwind)
    beam_divergence = height * np.deg2rad(beamwidth)
    power = 2 / 3
    a = (wind + beam_divergence) ** power
    b = (30 * wind + beam_divergence) ** power
    sigma_t = a * v_sigma / (b - a)
    return width - sigma_t ** 2


def calc_s(p, q, const, mu, beta, k):
    """ Help function for gamma-calculations """
    a = gamma(mu + p) / gamma(mu + (p - q))
    b = 1 / ((mu + 3.67) ** q)
    c = const * beta * k
    return a * b * c


def calc_dia(z, beta, mu, ray, k):
    """ Drizzle diameter calculation """
    p, q = 7, 3
    const = 2 / np.pi * ray / z
    s = calc_s(p, q, const, mu, beta, k)
    return (1 / s) ** (1 / (p - q))


def drizzle_solve(mie, width_ht):

    THRESHOLD = 1e-3

    Z = 10 ** ((vrs['Z'][:] - 180) / 10)

    beta = vrs['beta'][:]

    dz = np.median(np.diff(vrs['height'][:]))

    lu_medianD = mie['lu_medianD'][:]
    lu_u = mie['lu_u'][:]
    lu_k = mie['lu_k'][:]
    if is35:
        lu_width = mie['lu_width_35'][:]
        lu_ray = mie['lu_mie_ray_35'][:]
    else:
        lu_width = mie['lu_width_94'][:]
        lu_ray = mie['lu_mie_ray'][:]

    k = np.full((ntime, nalt), np.nan)
    mu = np.full((ntime, nalt), np.nan)
    mie_ray = np.full((ntime, nalt), np.nan)
    medianD = np.zeros((ntime, nalt))
    old_medianD = np.zeros((ntime, nalt))
    tab_mu = np.zeros((ntime, nalt))
    tab_oor = np.zeros((ntime, nalt))
    beta_corr = np.ones((ntime, nalt))
    tab_mie_ray = np.ones((ntime, nalt))
    init_k = 18.8
    tab_k = np.full((ntime, nalt), init_k)
    old_medianD[dind] = calc_dia(Z[dind], beta[dind] * init_k, 0.0, 1.0, 1.0)

    maxite = 10
    for i, j in zip(*dind):
        oldD = old_medianD[i, j]
        converged = False
        nite = 1
        while (not converged) and (nite < maxite):
            indD = nearest(lu_medianD, oldD)
            indmu = nearest(lu_width[:, indD], width_ht[i, j])
            tab_mu[i, j] = lu_u[indmu]
            tab_k[i, j] = lu_k[indmu, indD]
            tab_mie_ray[i, j] = lu_ray[indmu, indD]
            loopD = calc_dia(Z[i, j], beta[i, j], tab_mu[i, j], tab_mie_ray[i, j], tab_k[i, j])
            if (abs(loopD - oldD) < THRESHOLD):
                medianD[i, j] = loopD
                converged = True
            else:
                oldD = loopD
                nite = nite + 1

        beta_corr[i, (j + 1):] = beta_corr[i, (j + 1)] * np.exp(2 * tab_k[i, j] * beta[i, j] * dz)

    return (medianD, tab_mu, tab_k, tab_mie_ray, beta_corr)