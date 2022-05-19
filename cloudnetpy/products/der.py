"""Module for creating Cloudnet droplet effective radius using the Frisch et al. 2002 method."""
from collections import namedtuple
from typing import Optional

import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.categorize import atmos
from cloudnetpy.datasource import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import (
    CategorizeBits,
    ProductClassification,
    get_is_rain,
)

Parameters = namedtuple("Parameters", "ddBZ N dN sigma_x dsigma_x dQ")


def generate_der(
    categorize_file: str,
    output_file: str,
    uuid: Optional[str] = None,
    parameters: Optional[Parameters] = None,
) -> str:
    """Generates Cloudnet effective radius of liquid water droplets product acording
    to Frisch et al. 2002.

    This function calculates liquid droplet effective radius def using the Frisch method.
    In this method, def is calculated from radar reflectivity factor and microwave
    radiometer liquid water path. The results are written in a netCDF file.

    Args:
        categorize_file: Categorize file name.
        output_file: Output file name.
        uuid: Set specific UUID for the file.
        parameters: Tuple of specific fixed paramaters (ddBZ, N, dN, sigma_x, dsigma_x, dQ)
        used in Frisch approach.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_der
        >>> generate_der('categorize.nc', 'der.nc')
        >>>
        >>> from cloudnetpy.products.der import Parameters
        >>> params = Parameters(2.0, 100.0e6, 200.0e6, 0.25, 0.1, 5.0e-3)
        >>> generate_der('categorize.nc', 'der.nc', parameters=params)

    References:
        Frisch, S., Shupe, M., Djalalova, I., Feingold, G., & Poellot, M. (2002).
        The Retrieval of Stratus Cloud Droplet Effective Radius with Cloud Radars,
        Journal of Atmospheric and Oceanic Technology, 19(6), 835-842.
        Retrieved May 10, 2022, from
        https://doi.org/10.1175/1520-0426(2002)019%3C0835:TROSCD%3E2.0.CO;2

    """
    der_source = DerSource(categorize_file, parameters)
    droplet_classification = DropletClassification(categorize_file)
    der_source.append_der()
    der_source.append_retrieval_status(droplet_classification)
    date = der_source.get_date()
    attributes = output.add_time_attribute(REFF_ATTRIBUTES, date)
    attributes = _add_der_error_comment(attributes, der_source)
    output.update_attributes(der_source.data, attributes)
    uuid = output.save_product_file("der", der_source, output_file, uuid)
    der_source.close()
    return uuid


class DropletClassification(ProductClassification):
    """Class storing the information about different ice types.
    Child of ProductClassification().
    """

    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.is_mixed = self._find_mixed()
        self.is_droplet = self._find_droplet()
        self.is_ice = self._find_ice()

    def _find_droplet(self) -> np.ndarray:
        return self.category_bits["droplet"]

    def _find_mixed(self) -> np.ndarray:
        return self.category_bits["falling"] & self.category_bits["droplet"]

    def _find_ice(self) -> np.ndarray:
        return (
            self.category_bits["falling"]
            & self.category_bits["cold"]
            & ~self.category_bits["melting"]
            & ~self.category_bits["droplet"]
            & ~self.category_bits["insect"]
        )


class DerSource(DataSource):
    """Data container for effective radius calculations."""

    def __init__(self, categorize_file: str, parameters: Optional[Parameters] = None):
        super().__init__(categorize_file)
        self.is_rain = get_is_rain(categorize_file)
        self.categorize_bits = CategorizeBits(categorize_file)
        if parameters is None:
            # Default parameters from Frisch et al. 2002
            self.parameters = Parameters(2.0, 200.0e6, 200.0e6, 0.35, 0.1, 5.0e-3)
        else:
            self.parameters = parameters

    def append_der(self):
        """Estimate liquid droplet effective radius using Frisch et al. 2002."""

        params = self.parameters
        rho_l = 1000  # density of liquid water(kg m-3)

        var_x = params.sigma_x * params.sigma_x
        dheight = utils.mdiff(self.getvar("height"))

        Z = self.getvar("Z")
        Z = utils.db2lin(Z)
        dZ = ma.abs(utils.db2lin(params.ddBZ)) * Z

        lwp = self.getvar("lwp") * 1.0e-3  # g -> kg
        lwp[lwp < 0] = 0

        der = np.zeros(Z.shape)
        der_error = np.zeros(Z.shape)
        der_scaled = np.zeros(Z.shape)
        der_scaled_error = np.zeros(Z.shape)
        N_scaled = np.zeros(Z.shape)

        is_droplet = self.categorize_bits.category_bits["droplet"]
        liquid_bases = atmos.find_cloud_bases(is_droplet)
        liquid_tops = atmos.find_cloud_tops(is_droplet)

        for base, top in zip(zip(*np.where(liquid_bases)), zip(*np.where(liquid_tops))):
            ind_t = base[0]
            idx_layer = np.arange(base[1], top[1] + 1)

            if Z[ind_t, idx_layer].mask.all():
                continue

            integral = ma.sum(ma.sqrt(Z[ind_t, idx_layer])) * dheight

            # der formula (5)
            A = (Z[ind_t, idx_layer] / params.N) ** (1 / 6)
            B = ma.exp(-0.5 * var_x)
            der[ind_t, idx_layer] = 0.5 * A * B

            # der error formula (7)
            A = params.dN / (6 * params.N)
            B = params.sigma_x * params.dsigma_x
            C = dZ[ind_t, idx_layer] / (6 * Z[ind_t, idx_layer])
            der_error[ind_t, idx_layer] = der[ind_t, idx_layer] * ma.sqrt(A * A + B * B + C * C)

            # der scaled formula (6)
            A = Z[ind_t, idx_layer] ** (1 / 6) / (2 * lwp[ind_t] ** (1 / 3))
            B = (np.pi * rho_l / 6) ** (1 / 3)
            C = integral ** (1 / 3) * ma.exp(-2 * var_x)
            der_scaled[ind_t, idx_layer] = 1.0e-3 * A * B * C

            # der scaled formula (9)
            N_scaled[ind_t, idx_layer] = Z[ind_t, idx_layer] / (
                ((2 * der_scaled[ind_t, idx_layer]) / (ma.exp(-0.5 * var_x))) ** 6
            )
            A = dZ[ind_t, idx_layer] / (6 * Z[ind_t, idx_layer])
            B = 4 * params.sigma_x * params.dsigma_x
            C = params.dQ / (3 * lwp[ind_t])
            der_scaled_error[ind_t, idx_layer] = der_scaled[ind_t, idx_layer] * ma.sqrt(
                A * A + B * B + C * C
            )

        N_scaled = ma.masked_less_equal(ma.masked_invalid(N_scaled), 0.0) * 1.0e-6
        der = ma.masked_less_equal(ma.masked_invalid(der), 0.0) * 1.0e-3
        der_error = ma.masked_less_equal(ma.masked_invalid(der_error), 0.0) * 1.0e-3
        der_scaled = ma.masked_less_equal(ma.masked_invalid(der_scaled), 0.0) * 1.0e-3
        der_scaled_error = ma.masked_less_equal(ma.masked_invalid(der_scaled_error), 0.0) * 1.0e-3

        self.append_data(N_scaled, "N_scaled")
        self.append_data(der, "der")
        self.append_data(der_scaled, "der_scaled")
        self.append_data(der_error, "der_error")
        self.append_data(der_scaled_error, "der_scaled_error")

    def append_retrieval_status(self, droplet_classification: DropletClassification) -> None:
        """Returns information about the status of der retrieval."""
        is_retrieved = ~self.data["der"][:].mask
        is_mixed = droplet_classification.is_mixed
        is_ice = droplet_classification.is_ice
        is_rain = np.tile(self.is_rain, (is_retrieved.shape[1], 1)).T

        retrieval_status = np.zeros(is_retrieved.shape, dtype=int)
        retrieval_status[is_ice] = 4
        retrieval_status[is_retrieved] = 1
        retrieval_status[is_mixed * is_retrieved] = 2
        retrieval_status[is_rain * is_retrieved] = 3
        self.append_data(retrieval_status, "der_retrieval_status")


DEFINITIONS = {
    "der_retrieval_status": (
        "\n"
        "Value 0: No data: No cloud observed.\n"
        "Value 1: Reliable retrieval.\n"
        "Value 2: Mix of drops and ice: Droplets and ice crystals coexist within pixel.\n"
        "         Z may be biased by large crystals.\n"
        "Value 3: Precipitation in profile: Drizzle and rain affects LWP retrieval\n"
        "         of MWR but also the target reflectivity.\n"
        "Value 4: Surrounding ice: Less crucial! Ice crystals in the vicinity of a\n"
        "         droplet pixel may also bias its reflectivity.\n"
    )
}


COMMENTS = {
    "general": (
        "This dataset contains the effective cloud droplet radius calculated according to\n"
        "the approach presented by Frisch et al, 2002. It is based on either a direct\n"
        "relationship between Z and def for an assumed width and number concentration of the\n"
        "droplet concentration or on a method that relies only on the assumption of the\n"
        "width of the distribution by using radar and lidar to identify the liquid cloud\n"
        "base and top in each profile and scaling the liquid water content measured by microwave\n"
        "radiometer over the cloud."
    ),
    "der": (
        "This variable was calculated for the profiles where the categorization data has\n"
        "diagnosed that liquid water is present the cloud droplet effective radius is calculated\n"
        "after Frisch et al (2002), relating Z with def by assuming a lognormal size distribution\n"
        "its width and the number concentration of the cloud droplets."
    ),
    "der_scaled": (
        "This variable was calculated for the profiles where the\n"
        "categorization data has diagnosed that liquid water is present\n"
        "the cloud droplet effective radius is calculated after\n"
        "Frisch et al. (2002), relating Z with def by assuming a lognormal\n"
        "size distribution and its width. The number concentration required\n"
        "to represent the size distribution is derived by scaling the LWP\n "
        "measured with microwave radiometer over the observed (single) cloud layer."
    ),
    "der_scaled_error": (
        "This variable was calculated for the profiles where the categorization data"
    ),
    "N_scaled": (
        "From scaled Frisch method the cloud droplet number concentration can be derived."
    ),
}


def _add_der_error_comment(attributes: dict, der_source: DerSource) -> dict:
    params = der_source.parameters
    attributes["der_error"] = attributes["der_error"]._replace(
        comment="This variable is an estimate of the random error in effective\n"
        f"radius assuming an error in Z of ddBZ = {params.ddBZ} in N of dN = {params.dN}\n"
        f"and in the spectral width dsigma_x = {params.dsigma_x} and in the\n"
        f"LWP Q of {params.dQ} kg m-3."
    )
    return attributes


REFF_ATTRIBUTES = {
    "comment": COMMENTS["general"],
    "der": MetaData(
        long_name="Droplet effective radius",
        units="m",
        ancillary_variables="der_error",
        comment=COMMENTS["der"],
    ),
    "der_error": MetaData(
        long_name="Absolute error in droplet effective radius",
        units="m",
        comment="",
    ),
    "der_scaled": MetaData(
        long_name="Droplet effective radius (scaled to LWP)",
        units="m",
        ancillary_variables="der_scaled_error",
        comment=COMMENTS["der_scaled"],
    ),
    "der_scaled_error": MetaData(
        long_name="Absolute error in droplet effective radius (scaled to LWP)",
        units="m",
        comment=COMMENTS["der_scaled_error"],
    ),
    "N_scaled": MetaData(
        long_name="Cloud droplet number concentration",
        units="1",
        ancillary_variables="der_error der_scaled der_scaled_error",
        comment=COMMENTS["N_scaled"],
    ),
    "der_retrieval_status": MetaData(
        long_name="Droplet effective radius retrieval status",
        definition=DEFINITIONS["der_retrieval_status"],
        units="1",
    ),
}
