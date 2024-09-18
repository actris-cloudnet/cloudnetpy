"""Module for creating Cloudnet ice effective radius file using Z-T method."""

import numpy as np
from numpy import ma

from cloudnetpy import constants, output, utils
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.iwc import DEFINITIONS as IWC_DEFINITION
from cloudnetpy.products.product_tools import IceClassification, IceSource


def generate_ier(
    categorize_file: str,
    output_file: str,
    uuid: str | None = None,
) -> str:
    """Generates Cloudnet ice effective radius product.

    This function calculates ice particle effective radius using the Grieche
    et al. 2020 method which uses Hogan et al. 2006 to estimate ice water content
    and alpha from Delanoë et al. 2007. In this method, effective radius
    of ice particles is calculated from attenuated-corrected radar reflectivity
    and model temperature. The results are written in a netCDF file.

    Args:
        categorize_file: Categorize file name.
        output_file: Output file name.
        uuid: Set specific UUID for the file.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_ier
        >>> generate_ier('categorize.nc', 'ier.nc')

    References:
        Hogan, R. J., Mittermaier, M. P., & Illingworth, A. J. (2006). The Retrieval
        of Ice Water Content from Radar Reflectivity Factor and Temperature and Its
        Use in Evaluating a Mesoscale Model, Journal of Applied Meteorology and
        Climatology, 45(2), 301-317.
        from https://journals.ametsoc.org/view/journals/apme/45/2/jam2340.1.xml

        Delanoë, J., Protat, A., Bouniol, D., Heymsfield, A., Bansemer, A., & Brown,
        P. (2007). The Characterization of Ice Cloud Properties from Doppler Radar
        Measurements, Journal of Applied Meteorology and Climatology, 46(10),
        1682-1698.
        from https://journals.ametsoc.org/view/journals/apme/46/10/jam2543.1.xml

        Griesche, H. J., Seifert, P., Ansmann, A., Baars, H., Barrientos Velasco,
        C., Bühl, J., Engelmann, R., Radenz, M., Zhenping, Y., and Macke, A. (2020):
        Application of the shipborne remote sensing supersite OCEANET for
        profiling of Arctic aerosols and clouds during Polarstern cruise PS106,
        Atmos. Meas. Tech., 13, 5335–5358.
        from https://doi.org/10.5194/amt-13-5335-2020,

    """
    product = "ier"
    with IerSource(categorize_file, product) as ier_source:
        ice_classification = IceClassification(categorize_file)
        ier_source.append_icy_data(ice_classification)
        ier_source.convert_units()
        ier_source.append_status(ice_classification)
        ier_source.append_ier_error()
        date = ier_source.get_date()
        attributes = output.add_time_attribute(IER_ATTRIBUTES, date)
        attributes = _add_ier_comment(attributes, ier_source)
        output.update_attributes(ier_source.data, attributes)
        return output.save_product_file(product, ier_source, output_file, uuid)


class IerSource(IceSource):
    """Data container for ice effective radius calculations."""

    def convert_units(self) -> None:
        """Convert um to m."""
        self.data["ier"].data[:] /= 1e6

    def append_ier_error(self) -> None:
        error = ma.copy(self.data[f"{self.product}"][:])
        error = error * np.sqrt(0.4**2 + 0.4**2)
        self.append_data(error, f"{self.product}_error")


def _add_ier_comment(attributes: dict, ier: IerSource) -> dict:
    freq = utils.get_frequency(ier.wl_band)
    coeffs = ier.coefficients
    factor = np.round((coeffs[0] / 0.93), 3)
    attributes["ier"] = attributes["ier"]._replace(
        comment=f"This variable was calculated from the {freq}-GHz radar\n"
        f"reflectivity factor after correction for gaseous attenuation,\n"
        f"and temperature taken from a forecast model, using the following\n"
        f"empirical formula: log10(ier[m]) = ({coeffs[1]} * Z[dBZ] * T[degC]\n"
        f"+ {coeffs[3]} * Z[dBZ] + {coeffs[2]} * T[degC] + {coeffs[4]})\n"
        f"* 3 / (2 * {constants.RHO_ICE}[kg/m3]). In this formula Z is taken\n"
        "to be defined such that all frequencies of radar would measure the\n"
        "same Z in Rayleigh scattering ice. However, the radar is more likely\n"
        "to have been calibrated such that all frequencies would measure\n"
        "the same Z in Rayleigh scattering liquid cloud at 0 degrees C.\n"
        "The measured Z is therefore multiplied by\n"
        f" |K(liquid,0degC,{freq}GHz)|^2/0.93 = {factor} before applying\n"
        'this formula. The formula has been used where the "categorization"\n"'
        "data has diagnosed that the radar echo is due to ice, but note\n"
        "that in some cases supercooled drizzle will erroneously be identified\n"
        "as ice. Missing data indicates either that ice cloud was present but it was\n"
        "only detected by the lidar so its ice water content could not be estimated."
    )
    return attributes


COMMENTS = {
    "ier_error": (
        "Error in effective radius of ice particles due to error propagation,\n"
        "of ier = 3/(2 rho_i) IWC / alpha, using error for IWC and alpha as\n"
        "given in Hogan 2006."
    ),
    "ier_retrieval_status": (
        "This variable describes whether a retrieval was performed\n"
        "for each pixel, and its associated quality."
    ),
}

DEFINITIONS = {"ier_retrieval_status": IWC_DEFINITION["iwc_retrieval_status"]}

IER_ATTRIBUTES = {
    "ier": MetaData(
        long_name="Ice effective radius",
        units="m",
        ancillary_variables="ier_error",
    ),
    "ier_error": MetaData(
        long_name="Random error in ice effective radius",
        units="m",
        comment=COMMENTS["ier_error"],
    ),
    "ier_retrieval_status": MetaData(
        long_name="Ice effective radius retrieval status",
        comment=COMMENTS["ier_retrieval_status"],
        definition=DEFINITIONS["ier_retrieval_status"],
        units="1",
    ),
}
