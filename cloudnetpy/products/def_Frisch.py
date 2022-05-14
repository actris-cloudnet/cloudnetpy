"""Module for creating Cloudnet droplet effective radius using the Frisch et al. 2002 method."""
from collections import namedtuple
from typing import Optional, Tuple

import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.categorize.atmos import find_cloud_bases, find_cloud_tops
from cloudnetpy.datasource import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import (
    CategorizeBits,
    ProductClassification,
    get_is_rain,
)

Parameters = namedtuple("Parameters", "ddBZ N dN sigma_x dsigma_x dQ")


def generate_def_Frisch(
    categorize_file: str,
    output_file: str,
    uuid: Optional[str] = None,
    parameter: Optional[Tuple] = None,
) -> str:
    """Generates Cloudnet effective radius of liquid water droplets product acording to Frisch et al. 2002.

    This function calculates liquid droplet effective radius def using the Frisch method.
    In this method, def is calculated from radar reflectivity factor and microwave
    radiometer liquid water path. The results are written in a netCDF file.

    Args:
        categorize_file: Categorize file name.
        output_file: Output file name.
        uuid: Set specific UUID for the file.
        parameter: Tuple of specific fixed paramaters (ddBZ, N, dN, sigma_x, dsigma_x, dQ) used in Frisch approach.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_def_Frisch
        >>> generate_def_Frisch('categorize.nc', 'def_Frisch.nc')

    References:
        Frisch, S., Shupe, M., Djalalova, I., Feingold, G., & Poellot, M. (2002).
        The Retrieval of Stratus Cloud Droplet Effective Radius with Cloud Radars,
        Journal of Atmospheric and Oceanic Technology, 19(6), 835-842.
        Retrieved May 10, 2022, from https://doi.org/10.1175/1520-0426(2002)019%3C0835:TROSCD%3E2.0.CO;2

    """
    def_source = DefSource(categorize_file)
    droplet_classification = DropletClassification(categorize_file)

    def_source.append_parameter(parameter)
    def_source.append_def_Frisch()
    def_source.append_retrieval_status(droplet_classification)

    date = def_source.get_date()
    attributes = output.add_time_attribute(REFF_ATTRIBUTES, date)

    output.update_attributes(def_source.data, attributes)
    uuid = output.save_product_file("def_Frisch", def_source, output_file, uuid)
    def_source.close()
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


class DefSource(DataSource):

    """Data container for effective radius calculations."""

    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.Z = self.getvar("Z")

        # from lwp source
        self.lwp = self.getvar("lwp")
        self.lwp_error = self.getvar("lwp_error")
        self.lwp[self.lwp < 0] = 0
        self.is_rain = get_is_rain(categorize_file)
        self.dheight = utils.mdiff(self.getvar("height"))
        self.categorize_bits = CategorizeBits(categorize_file)

    def append_parameter(self, parameter):
        """Define constant parameters for the Frisch method.

        Returns:
            Parameters for droplet effective radius computation.
        """

        # Default fixed parameters from Frisch et al. 2002
        self.params = Parameters(2.0, 200.0e6, 200.0e6, 0.35, 0.1, 5.0e-3)

        if parameter is not None:
            self.params = Parameters(*parameter)

    def append_def_Frisch(self):

        """Estimate liquid droplet effective radius using Frisch et al. 2002."""

        params = self.params
        var_x = params.sigma_x * params.sigma_x

        ntime, nrange = self.Z.shape

        is_droplet = self.categorize_bits.category_bits["droplet"]
        is_some_liquid = np.any(is_droplet, axis=1)

        # density of liquid water(kg m-3)
        rho_l = 1000
        pi = np.pi

        # convert to linear units
        Z = utils.db2lin(self.Z)
        dZ = ma.abs(utils.db2lin(params.ddBZ)) * Z

        lwp = self.lwp
        lwp_error = self.lwp_error
        lwp *= 1.0e-3  # convert to kg m-2
        lwp_error *= 1.0e-3

        liquid_bases = find_cloud_bases(is_droplet)
        liquid_tops = find_cloud_tops(is_droplet)

        def_Frisch = np.zeros((ntime, nrange))
        def_error = np.zeros((ntime, nrange))
        def_scaled = np.zeros((ntime, nrange))
        def_scaled_error = np.zeros((ntime, nrange))
        N_scaled = np.zeros((ntime, nrange))

        # loop over all profiles
        for ind_t in range(ntime):

            if not is_some_liquid[ind_t]:
                continue

            for base, top in zip(
                zip(*np.where(liquid_bases[ind_t, :])), zip(*np.where(liquid_tops[ind_t, :]))
            ):
                idx_layer = np.arange(base[0], top[0] + 1)

                # if all values of Z between base and top are NAN, contine
                if Z[ind_t, idx_layer].mask.all():
                    continue

                integral = ma.sum(ma.sqrt(Z[ind_t, idx_layer])) * self.dheight

                # def_Frisch formula (5)
                A = (Z[ind_t, idx_layer] / params.N) ** (1 / 6)
                B = ma.exp(-0.5 * var_x)
                def_Frisch[ind_t, idx_layer] = 0.5 * A * B

                # def_Frisch error formula (7)
                A = params.dN / (6 * params.N)
                B = params.sigma_x * params.dsigma_x
                C = dZ[ind_t, idx_layer] / (6 * Z[ind_t, idx_layer])
                def_error[ind_t, idx_layer] = def_Frisch[ind_t, idx_layer] * ma.sqrt(
                    A * A + B * B + C * C
                )

                # def_Frisch scaled formula (6)
                A = Z[ind_t, idx_layer] ** (1 / 6) / (2 * lwp[ind_t] ** (1 / 3))
                B = (pi * rho_l / 6) ** (1 / 3)
                C = integral ** (1 / 3) * ma.exp(-2 * var_x)
                def_scaled[ind_t, idx_layer] = 1.0e-3 * A * B * C

                # def_Frisch scaled formula (9)
                N_scaled[ind_t, idx_layer] = Z[ind_t, idx_layer] / (
                    ((2 * def_scaled[ind_t, idx_layer]) / (ma.exp(-0.5 * var_x))) ** 6
                )
                A = dZ[ind_t, idx_layer] / (6 * Z[ind_t, idx_layer])
                B = 4 * params.sigma_x * params.dsigma_x
                C = params.dQ / (3 * lwp[ind_t])
                def_scaled_error[ind_t, idx_layer] = def_scaled[ind_t, idx_layer] * ma.sqrt(
                    A * A + B * B + C * C
                )

        N_scaled = ma.masked_less_equal(ma.masked_invalid(N_scaled), 0.0) * 1.0e-6
        def_Frisch = ma.masked_less_equal(ma.masked_invalid(def_Frisch), 0.0) * 1.0e-3
        def_error = ma.masked_less_equal(ma.masked_invalid(def_error), 0.0) * 1.0e-3
        def_scaled = ma.masked_less_equal(ma.masked_invalid(def_scaled), 0.0) * 1.0e-3
        def_scaled_error = ma.masked_less_equal(ma.masked_invalid(def_scaled_error), 0.0) * 1.0e-3

        self.append_data(N_scaled, "N_scaled_Frisch")
        self.append_data(def_Frisch, "def_Frisch")
        self.append_data(def_scaled, "def_scaled_Frisch")
        self.append_data(def_error, "def_error_Frisch")
        self.append_data(def_scaled_error, "def_scaled_error_Frisch")

    def append_retrieval_status(self, droplet_classification: DropletClassification) -> None:
        """Returns information about the status of def_Frisch retrieval."""
        is_retrieved = ~self.data["def_Frisch"][:].mask
        is_mixed = droplet_classification.is_mixed
        is_ice = droplet_classification.is_ice
        is_rain = np.tile(self.is_rain, (is_retrieved.shape[1], 1)).T

        retrieval_status = np.zeros(is_retrieved.shape)
        retrieval_status[is_ice] = 4
        retrieval_status[is_retrieved] = 1
        retrieval_status[is_mixed * is_retrieved] = 2
        retrieval_status[is_rain * is_retrieved] = 3
        self.append_data(retrieval_status, "def_Frisch_retrieval_status")


DEFINITIONS = {
    "def_Frisch_retrieval_status": (
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
        "the approach presented by Frisch et al, 2002. It is based on either a direct relations\n"
        "ship between Z and def for an assumed width and number concentration of the\n"
        "droplet concentration or on a method that relies only on the assumption of the \n"
        "width of the distribution by using radar and lidar to identify the liquid cloud\n"
        "base and top in each profile and scaling the liquid water content measured by microwave\n"
        "radiometer over the cloud."
    ),
    "def": (
        "This variable was calculated for the profiles where the \n"
        "categorization data has diagnosed that liquid water is present \n"
        "the cloud droplet effective radius is calculated after \n"
        "Frisch et al (2002), relating Z with def by assuming a \n"
        "lognormal size distribution its width and the number \n"
        "concentration of the cloud droplets."
    ),
    "def_error": (
        "This variable is an estimate of the random error in effective \n"
        "radius assuming an error in Z of ddBZ = 2.0 in N of dN = 200.0e6 \n"
        "and in the spectral width dsigma_x = 0.1 and in the LWP Q of 5e-3 kg m-3."
    ),
    "def_scaled": (
        "This variable was calculated for the profiles where the \n"
        "categorization data has diagnosed that liquid water is present \n"
        "the cloud droplet effective radius is calculated after \n"
        "Frisch et al. (2002), relating Z with def by assuming a lognormal \n"
        "size distribution and its width. The number concentration required \n"
        "to represent the size distribution is derived by scaling the LWP \n "
        "measured with microwave radiometer over the observed (single) cloud layer."
    ),
    "def_scaled_error": (
        "This variable was calculated for the profiles where the categorization data"
    ),
    "N_scaled": (
        "From scaled Frisch method the cloud droplet number concentration can be derived."
    ),
}

REFF_ATTRIBUTES = {
    "comment": COMMENTS["general"],
    "def_Frisch": MetaData(
        long_name="Effective radius",
        units="m",
        ancillary_variables="def_error",
        comment=COMMENTS["def"],
    ),
    "def_error_Frisch": MetaData(
        long_name="Absolute error in effective radius",
        units="m",
        comment=COMMENTS["def_error"],
    ),
    "def_scaled_Frisch": MetaData(
        long_name="Effective radius (scaled to LWP)",
        units="m",
        ancillary_variables="def_scaled_error",
        comment=COMMENTS["def_scaled"],
    ),
    "def_scaled_error_Frisch": MetaData(
        long_name="Absolute error in effective radius (scaled to LWP)",
        units="m",
        comment=COMMENTS["def_scaled_error"],
    ),
    "N_scaled_Frisch": MetaData(
        long_name="Cloud droplet number concentration",
        units="1",
        ancillary_variables="def_error def_scaled def_scaled_error",
        comment=COMMENTS["N_scaled"],
    ),
    "def_Frisch_retrieval_status": MetaData(
        long_name="Droplet effective radius retrieval status",
        definition=DEFINITIONS["def_Frisch_retrieval_status"],
        units="1",
    ),
}
