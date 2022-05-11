"""Module for creating Cloudnet ice water content file using Z-T method."""
from collections import namedtuple
from typing import Optional

import numpy as np
from cloudnetpy import output, utils
from cloudnetpy.datasource import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import ProductClassification, get_is_rain, CategorizeBits
from numpy import ma

Parameters = namedtuple("Parameters", "ddBZ N dN sigmax dsigmax dQ")


def generate_reff_Frisch(categorize_file: str, output_file: str, uuid: Optional[str] = None) -> str:
    """Generates Cloudnet effective radius of liquid water droplets product acording to Frisch et al. 2002.

    This function calculates liquid droplet effective radius reff using the Frisch method.
    In this method, reff is calculated from radar reflectivity factor and microwave
    radiometer liquid water path. The results are written in a netCDF file.

    Args:
        categorize_file: Categorize file name.
        output_file: Output file name.
        uuid: Set specific UUID for the file.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_reff_Frisch
        >>> generate_reff_Frisch('categorize.nc', 'reff_Frisch.nc')

    References:
        Frisch, S., Shupe, M., Djalalova, I., Feingold, G., & Poellot, M. (2002).
        The Retrieval of Stratus Cloud Droplet Effective Radius with Cloud Radars,
        Journal of Atmospheric and Oceanic Technology, 19(6), 835-842.
        Retrieved May 10, 2022, from https://doi.org/10.1175/1520-0426(2002)019%3C0835:TROSCD%3E2.0.CO;2

    """
    reff_source = ReffSource(categorize_file)
    droplet_classification = DropletClassification(categorize_file)

    reff_source.append_reff_Frisch()
    reff_source.append_retrieval_status(droplet_classification)

    date = reff_source.get_date()
    attributes = output.add_time_attribute(REFF_ATTRIBUTES, date)

    output.update_attributes(reff_source.data, attributes)
    uuid = output.save_product_file("reff-Frisch", reff_source, output_file, uuid)
    reff_source.close()
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
        return  self.category_bits["droplet"]

    def _find_mixed(self) -> np.ndarray:
        return (
                self.category_bits["falling"]
                & self.category_bits["droplet"]
        )

    def _find_ice(self) -> np.ndarray:
        return (
            self.category_bits["falling"]
            & self.category_bits["cold"]
            & ~self.category_bits["melting"]
            & ~self.category_bits["droplet"]
            & ~self.category_bits["insect"]
        )

class ReffSource(DataSource):

    """Data container for effective radius calculations."""

    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.wl_band = utils.get_wl_band(float(self.getvar("radar_frequency")))
        self.params = self._get_reff_params()
        self.Z = self._get_reflectivity_factor()

        # from lwp source
        self.lwp = self.getvar("lwp")
        self.lwp_error = self.getvar("lwp_error")
        self.lwp[self.lwp < 0] = 0
        self.is_rain = get_is_rain(categorize_file)
        self.dheight = utils.mdiff(self.getvar("height"))
        self.categorize_bits = CategorizeBits(categorize_file)


    def append_reff_Frisch(self):

        """ Estimate liquid droplet effective radius using Frisch et al. 2002.

        """

        ddBZ, N, dN, sigma_x, dsigma_x, dQ = [self.params[int(_)] for _ in range(len(self.params))]
        var_x = sigma_x*sigma_x

        ntime, nrange = self.Z.shape

        is_droplet = self.categorize_bits.category_bits['droplet']
        is_some_liquid = np.any(is_droplet, axis=1)

        # density of liquid water(kg m-3)
        rho_l = 1000
        pi = np.pi

        # convert to linear units
        Z = 10 ** (self.Z.copy() / 10.)
        dZ = np.abs(10. ** (ddBZ / 10)) * Z

        lwp = self.lwp
        lwp_error = self.lwp_error
        # convert to kg m-2
        if 'kg' not in self.dataset.variables['lwp'].units:
            lwp *= 1.0e-3
            lwp_error *= 1.0e-3

        reff = np.zeros((ntime, nrange))
        reff_error = np.zeros((ntime, nrange))
        reff_scaled = np.zeros((ntime, nrange))
        reff_scaled_error = np.zeros((ntime, nrange))
        N_scaled = np.zeros((ntime, nrange))
        N_scaled_error = np.zeros((ntime, nrange))

        # loop over all profiles
        for ind_t in range(ntime):

            if not is_some_liquid[ind_t]:
                continue

            diff_droplet_bit = np.diff(np.concatenate([[False], is_droplet[ind_t, :], [False]]).astype(int))
            n_liquid_layers = np.count_nonzero(diff_droplet_bit)//2

            liquid_tops = np.argwhere(diff_droplet_bit == - 1).reshape(-1) - 1
            liquid_bases = np.argwhere(diff_droplet_bit == 1).reshape(-1)

            for ind_layer in range(n_liquid_layers):
                idx_layer = np.arange(liquid_bases[ind_layer], liquid_tops[ind_layer]+1)

                # if all values of Z between base and top are NAN, contine
                if Z[ind_t, idx_layer].mask.all():
                    continue

                integral = ma.sum(ma.sqrt(Z[ind_t, idx_layer])) * self.dheight
                integral_error = ma.sum(np.sqrt(Z[ind_t, idx_layer] + dZ[ind_t, idx_layer])) * self.dheight

                # reff formula (5)
                A = (Z[ind_t, idx_layer] / N) ** (1/6)
                B = ma.exp(-0.5 * var_x)
                reff[ind_t, idx_layer] = 0.5 * A * B

                # reff error formula (7)
                A = dN / (6*N)
                B = sigma_x * dsigma_x
                C = dZ[ind_t, idx_layer] / (6 * Z[ind_t, idx_layer])
                reff_error[ind_t, idx_layer] = reff[ind_t, idx_layer] * ma.sqrt(A*A + B*B + C*C)

                # reff scaled formula (6)
                A = Z[ind_t, idx_layer] ** (1/6) / (2 * lwp[ind_t] ** (1/3))
                B = (pi * rho_l / 6) ** (1 / 3)
                C = integral ** (1 / 3) * ma.exp(- 2 * var_x)
                reff_scaled[ind_t, idx_layer] = 1.0e-3 * A * B * C

                # reff scaled formula (8) and (9)
                N_scaled[ind_t, idx_layer] = Z[ind_t, idx_layer] / (((2 * reff_scaled[ind_t, idx_layer]) / (ma.exp(-0.5 * var_x))) ** 6)
                A = dZ[ind_t, idx_layer] / (6*Z[ind_t, idx_layer])
                B = integral_error ** (1/3) / integral ** (1/3)
                C = 4*sigma_x*dsigma_x
                D = dQ / (3*lwp[ind_t])
                #reff_scaled_error[ind_t, idx_layer] = reff_scaled[ind_t, idx_layer] * ma.sqrt(A*A + B*B + C*C + D*D) # (8)
                reff_scaled_error[ind_t, idx_layer] = reff_scaled[ind_t, idx_layer] * ma.sqrt(A*A + C*C + D*D) # (9)

        N_scaled = ma.masked_less_equal(ma.masked_invalid(N_scaled), 0.0) * 1.0e-6
        reff = ma.masked_less_equal(ma.masked_invalid(reff), 0.0)* 1.0e-3
        reff_error = ma.masked_less_equal(ma.masked_invalid(reff_error), 0.0)* 1.0e-3
        reff_scaled = ma.masked_less_equal(ma.masked_invalid(reff_scaled), 0.0)* 1.0e-3
        reff_scaled_error = ma.masked_less_equal(ma.masked_invalid(reff_scaled_error), 0.0)* 1.0e-3

        self.append_data(N_scaled, "N_scaled_Frisch")
        self.append_data(reff, "reff_Frisch")
        self.append_data(reff_scaled, "reff_scaled_Frisch")
        self.append_data(reff_error, "reff_error_Frisch")
        self.append_data(reff_scaled_error, "reff_scaled_error_Frisch")

    def append_retrieval_status(self, droplet_classification: DropletClassification) -> None:
        """Returns information about the status of reff retrieval."""
        is_retrieved = ~self.data["reff_Frisch"][:].mask
        is_mixed = droplet_classification.is_mixed
        is_ice = droplet_classification.is_ice
        is_rain = np.tile(self.is_rain, (is_retrieved.shape[1], 1)).T

        retrieval_status = np.zeros(is_retrieved.shape)
        retrieval_status[is_ice] = 4
        retrieval_status[is_retrieved] = 1
        retrieval_status[is_mixed * is_retrieved] = 2
        retrieval_status[is_rain * is_retrieved] = 3
        self.append_data(retrieval_status, "reff_Frisch_retrieval_status")


    def _get_reff_params(self) -> Parameters:
        """ Define constant parameters for the Frisch method.

        Returns:
            Parameters for
                - ddBZ in [dB]
                - N in [m-3]
                - dN [??]
                - sigma_x [??]
                - dsigma_x [??]
                - dQ is the change in microwave radiometer–derived integrated liquid water through the depth of the cloud over time [kg m-2 s-1]

        """
        if 'punta' in self.dataset.location.lower():
            # 20190812 Patric Seifert - New Values for Punta Arenas
            # Fixed Parameters for South Atlantic from Martin et al., 1994 (Fig. 3a)
            return Parameters(2.0, 130.e6, 200.e6, 0.29, 0.1, 5.e-3)

        ##20190812 Patric - Default values until 20190812
        # Fixed Parameters from Frisch et al. 2002
        return Parameters(2.0, 200.e6, 200.e6, 0.35, 0.1, 5.e-3)


    def _get_reflectivity_factor(self) -> np.ndarray:
        """Returns interpolated radar reflectivity factor in dBZ."""
        return self.getvar("Z")

DEFINITIONS = {
    "reff_retrieval_status": (
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
        "ship between Z and reff for an assumed width and number concentration of the\n"
        "droplet concentration or on a method that relies only on the assumption of the \n"
        "width of the distribution by using radar and lidar to identify the liquid cloud\n"
        "base and top in each profile and scaling the liquid water content measured by microwave\n"
        "radiometer over the cloud."
    ),
    "reff": (
        "This variable was calculated for the profiles where the \n"
        "categorization data has diagnosed that liquid water is present \n"
        "the cloud droplet effective radius is calculated after \n"
        "Frisch et al (2002), relating Z with reff by assuming a \n"
        "lognormal size distribution its width and the number \n"
        "concentration of the cloud droplets."
    ),
    "reff_error": (
        "This variable is an estimate of the random error in effective \n"
        "radius assuming an error in Z of ddBZ = 2.0 in N of dN = 200.0e6 \n"
        "and in the spectral width dsigma_x = 0.1 and in the LWP Q of 5e-3 kg m-3."
    ),
    "reff_scaled": (
        "This variable was calculated for the profiles where the \n"
        "categorization data has diagnosed that liquid water is present \n"
        "the cloud droplet effective radius is calculated after \n"
        "Frisch et al. (2002), relating Z with reff by assuming a lognormal \n"
        "size distribution and its width. The number concentration required \n"
        "to represent the size distribution is derived by scaling the LWP \n "
        "measured with microwave radiometer over the observed (single) cloud layer."
    ),
    "reff_scaled_error": (
        "This variable was calculated for the profiles where the categorization data"
    ),
    "N_scaled": (
        "From scaled Frisch method the cloud droplet number concentration can be derived."
    ),
    "N_scaled_error": (
        "The absolute error of the droplet number concentration derived from \n"
        "scaled droplet effective radius."
    )
}

REFF_ATTRIBUTES = {
    "reff_Frisch": MetaData(
        long_name="Effective radius",
        units="m",
        ancillary_variables="reff_error",
    ),
    "reff_error_Frisch": MetaData(
        long_name="Absolute error in effective radius",
        units="m",
    ),
    "reff_scaled_Frisch": MetaData(
        long_name="Effective radius (scaled to LWP)",
        units="m",
        ancillary_variables="reff_scaled_error",
    ),
    "reff_scaled_error_Frisch": MetaData(
        long_name="Absolute error in effective radius (scaled to LWP)",
        units="m",
    ),
    "N_scaled_Frisch": MetaData(
        long_name="Cloud droplet number concentration",
        units="1",
        ancillary_variables="reff_error reff_scaled reff_scaled_error",
    ),
    "N_scaled_error_Frisch": MetaData(
        long_name="Absolute error in cloud droplet number concentration",
        units="1",
    ),

}
