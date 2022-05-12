"""Module for creating Cloudnet drizzle product.
"""
from typing import Optional

import numpy as np
from numpy import ma
from scipy.special import gamma

from cloudnetpy import output, utils
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.drizzle_error import get_drizzle_error
from cloudnetpy.products.drizzle_tools import (
    DrizzleClassification,
    DrizzleSolver,
    DrizzleSource,
    SpectralWidth,
)


def generate_drizzle(categorize_file: str, output_file: str, uuid: Optional[str] = None) -> str:
    """Generates Cloudnet drizzle product.

    This function calculates different drizzle properties from
    cloud radar and lidar measurements. The results are written in a netCDF file.

    Args:
        categorize_file: Categorize file name.
        output_file: Output file name.
        uuid: Set specific UUID for the file.

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
    with DrizzleSource(categorize_file) as drizzle_source:
        drizzle_class = DrizzleClassification(categorize_file)
        spectral_width = SpectralWidth(categorize_file)
        drizzle_solver = DrizzleSolver(drizzle_source, drizzle_class, spectral_width)
        derived_products = DrizzleProducts(drizzle_source, drizzle_solver)
        errors = get_drizzle_error(drizzle_source, drizzle_solver)
        retrieval_status = RetrievalStatus(drizzle_class)
        results = {**drizzle_solver.params, **derived_products.derived_products, **errors}
        results = _screen_rain(results, drizzle_class)
        results["drizzle_retrieval_status"] = retrieval_status.retrieval_status
        _append_data(drizzle_source, results)
        date = drizzle_source.get_date()
        attributes = output.add_time_attribute(DRIZZLE_ATTRIBUTES, date)
        output.update_attributes(drizzle_source.data, attributes)
        uuid = output.save_product_file("drizzle", drizzle_source, output_file, uuid)
    return uuid


class DrizzleProducts:
    """Calculates additional quantities from the drizzle properties.

    Args:
        drizzle_source: The :class:`DrizzleSource` instance.
        drizzle_solver: The :class:`DrizzleSolver` instance.

    Attributes:
        derived_products (dict): Dictionary containing derived drizzle products:
            'drizzle_N', 'drizzle_lwc', 'drizzle_lwf', 'v_drizzle', 'v_air'.

    """

    def __init__(self, drizzle_source: DrizzleSource, drizzle_solver: DrizzleSolver):
        self._data = drizzle_source
        self._params = drizzle_solver.params
        self._ind_drizzle, self._ind_lut = self._find_indices()
        self.derived_products = self._calc_derived_products()

    def _find_indices(self):
        drizzle_ind = np.where(self._params["Do"])
        ind_mu = np.searchsorted(self._data.mie["mu"], self._params["mu"][drizzle_ind])
        ind_dia = np.searchsorted(self._data.mie["Do"], self._params["Do"][drizzle_ind])
        n_widths, n_dia = len(self._data.mie["mu"]), len(self._data.mie["Do"])
        ind_mu[ind_mu >= n_widths] = n_widths - 1
        ind_dia[ind_dia >= n_dia] = n_dia - 1
        return drizzle_ind, (ind_mu, ind_dia)

    def _calc_derived_products(self):
        density = self._calc_density()
        lwc = self._calc_lwc()
        lwf = self._calc_lwf(lwc)
        v_drizzle = self._calc_fall_velocity()
        v_air = self._calc_v_air(v_drizzle)
        return {
            "drizzle_N": density,
            "drizzle_lwc": lwc,
            "drizzle_lwf": lwf,
            "v_drizzle": v_drizzle,
            "v_air": v_air,
        }

    def _calc_density(self):
        """Calculates drizzle number density (m-3)."""
        a = self._data.z * 3.67**6
        b = self._params["Do"] ** 6
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def _calc_lwc(self):
        """Calculates drizzle liquid water content (kg m-3)"""
        rho_water = 1000
        dia, mu, s = [self._params.get(key) for key in ("Do", "mu", "S")]
        gamma_ratio = gamma(4 + mu) / gamma(3 + mu) / (3.67 + mu)
        return rho_water / 3 * self._data.beta * s * dia * gamma_ratio

    def _calc_lwf(self, lwc_in):
        """Calculates drizzle liquid water flux."""
        flux = ma.copy(lwc_in)
        flux[self._ind_drizzle] *= (
            self._data.mie["lwf"][self._ind_lut] * self._data.mie["termv"][self._ind_lut[1]]
        )
        return flux

    def _calc_fall_velocity(self):
        """Calculates drizzle droplet fall velocity (m s-1)."""
        velocity = np.zeros_like(self._params["Do"])
        velocity[self._ind_drizzle] = -self._data.mie["v"][self._ind_lut]
        return velocity

    def _calc_v_air(self, droplet_velocity):
        """Calculates vertical air velocity."""
        velocity = -np.copy(droplet_velocity)
        velocity[self._ind_drizzle] += self._data.v[self._ind_drizzle]
        return velocity


class RetrievalStatus:
    """Estimates the status of drizzle retrievals.

    Args:
        drizzle_class: The :class:`DrizzleClassification` instance.

    Attributes:
        drizzle_class: The :class:`DrizzleClassification` instance.
        retrieval_status (ndarray): 2D array containing drizzle retrieval status information.
    """

    def __init__(self, drizzle_class: DrizzleClassification):
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


def _screen_rain(results: dict, classification: DrizzleClassification):
    """Removes rainy profiles from drizzle variables.."""
    for key in results.keys():
        if not utils.isscalar(results[key]):
            results[key][classification.is_rain, :] = 0
    return results


def _append_data(drizzle_data: DrizzleSource, results: dict):
    """Save retrieved fields to the drizzle_data object."""
    for key, value in results.items():
        if key != "drizzle_retrieval_status":
            value = ma.masked_where(value == 0, value)
        drizzle_data.append_data(value, key)


DRIZZLE_ATTRIBUTES = {
    "drizzle_N": MetaData(
        long_name="Drizzle number concentration",
        units="m-3",
        ancillary_variables="drizzle_N_error drizzle_N_bias",
    ),
    "drizzle_N_error": MetaData(
        long_name="Random error in drizzle number concentration", units="dB"
    ),
    "drizzle_N_bias": MetaData(
        long_name="Possible bias in drizzle number concentration",
        units="dB",
    ),
    "drizzle_lwc": MetaData(
        long_name="Drizzle liquid water content",
        units="kg m-3",
        ancillary_variables="drizzle_lwc_error drizzle_lwc_bias",
    ),
    "drizzle_lwc_error": MetaData(
        long_name="Random error in drizzle liquid water content",
        units="dB",
    ),
    "drizzle_lwc_bias": MetaData(
        long_name="Possible bias in drizzle liquid water content",
        units="dB",
    ),
    "drizzle_lwf": MetaData(
        long_name="Drizzle liquid water flux",
        units="kg m-2 s-1",
        ancillary_variables="drizzle_lwf_error drizzle_lwf_bias",
    ),
    "drizzle_lwf_error": MetaData(
        long_name="Random error in drizzle liquid water flux",
        units="dB",
    ),
    "drizzle_lwf_bias": MetaData(
        long_name="Possible bias in drizzle liquid water flux",
        units="dB",
    ),
    "v_drizzle": MetaData(
        long_name="Drizzle droplet fall velocity",  # TODO: should it include 'terminal' ?
        units="m s-1",
        ancillary_variables="v_drizzle_error v_drizzle_bias",
        comment="Positive values are towards the ground.",
    ),
    "v_drizzle_error": MetaData(
        long_name="Random error in drizzle droplet fall velocity", units="dB"
    ),
    "v_drizzle_bias": MetaData(
        long_name="Possible bias in drizzle droplet fall velocity",
        units="dB",
    ),
    "v_air": MetaData(
        long_name="Vertical air velocity",
        units="m s-1",
        ancillary_variables="v_air_error",
        comment="Positive values are towards the sky.",
    ),
    "v_air_error": MetaData(long_name="Random error in vertical air velocity", units="dB"),
    "Do": MetaData(
        long_name="Drizzle median diameter", units="m", ancillary_variables="Do_error Do_bias"
    ),
    "Do_error": MetaData(
        long_name="Random error in drizzle median diameter",
        units="dB",
    ),
    "Do_bias": MetaData(
        long_name="Possible bias in drizzle median diameter",
        units="dB",
    ),
    "mu": MetaData(
        long_name="Drizzle droplet size distribution shape parameter",
        ancillary_variables="mu_error",
        units="1",
    ),
    "mu_error": MetaData(
        long_name="Random error in drizzle droplet size distribution shape parameter",
        units="dB",
    ),
    "S": MetaData(
        long_name="Lidar backscatter-to-extinction ratio", ancillary_variables="S_error", units="sr"
    ),
    "S_error": MetaData(
        long_name="Random error in lidar backscatter-to-extinction ratio", units="dB"
    ),
    "beta_corr": MetaData(long_name="Lidar backscatter correction factor", units="1"),
    "drizzle_retrieval_status": MetaData(long_name="Drizzle parameter retrieval status", units="1"),
}
