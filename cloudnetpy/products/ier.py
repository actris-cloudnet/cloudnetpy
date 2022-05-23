"""Module for creating Cloudnet ice effective radius file using Z-T method."""
from typing import Optional

import numpy as np
from numpy import ma

from cloudnetpy import constants, output, utils
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import IceClassification, IceSource


def generate_ier(categorize_file: str, output_file: str, uuid: Optional[str] = None) -> str:
    """Generates Cloudnet ice effective radius product.

    This function calculates ice particle effective radius using the Grieche et al. 2020 method
    which uses Hogan et al. 2006 to estimate ice water content and alpha from Delanoë et al. 2007.
    In this method, effective radius of ice particles is calculated from attenuated-corrected
    radar reflectivity and model temperature. The results are written in a netCDF file.

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
        ier_source.append_main_variable_including_rain(ice_classification)
        ier_source.append_main_variable(ice_classification)
        ier_source.append_status(ice_classification)
        ier_source.append_ier_error(ice_classification)
        date = ier_source.get_date()
        attributes = output.add_time_attribute(IER_ATTRIBUTES, date)
        attributes = _add_ier_comment(attributes, ier_source)
        output.update_attributes(ier_source.data, attributes)
        uuid = output.save_product_file(product, ier_source, output_file, uuid)
    return uuid


class IerSource(IceSource):
    """Data container for ice effective radius calculations."""

    def append_ier_error(self, ice_classification: IceClassification) -> None:
        error = ma.copy(self.data[f"{self.product}_inc_rain"][:])
        error[ice_classification.ice_above_rain] = ma.masked
        error = error * np.sqrt(0.4**2 + 0.4**2)
        self.append_data(error, f"{self.product}_error")


def _add_ier_comment(attributes: dict, ier: IerSource) -> dict:
    freq = utils.get_frequency(ier.wl_band)
    coeffs = ier.coefficients
    factor = np.round((coeffs[0] / 0.93), 3)
    attributes["ier"] = attributes["ier"]._replace(
        comment=f"This variable was calculated from the {freq}-GHz radar reflectivity factor\n"
        f"after correction for gaseous attenuation, and temperature taken from a forecast model,\n"
        f"using the following empirical formula: log10(ier[m]) =\n"
        f"({coeffs[1]} * Z[dBZ] * T[degC] + {coeffs[3]} * Z[dBZ]\n"
        f"+ {coeffs[2]} * T[degC] + {coeffs[4]}) * 3 / (2 * {constants.RHO_ICE}[kg/m3]).\n"
        "In this formula Z is taken to be defined such that all frequencies of radar would\n"
        "measure the same Z in Rayleigh scattering ice. However, the radar is more likely to\n"
        "have been calibrated such that all frequencies would measure the same Z in Rayleigh\n"
        "scattering liquid cloud at 0 degrees C. The measured Z is therefore multiplied by\n"
        f"|K(liquid,0degC,{freq}GHz)|^2/0.93 = {factor} before applying this formula.\n"
        'The formula has been used where the "categorization" data has diagnosed that the radar\n"'
        "echo is due to ice, but note that in some cases supercooled drizzle will erroneously be\n"
        "identified as ice. Missing data indicates either that ice cloud was present but it was\n"
        "only detected by the lidar so its ice water content could not be estimated, or than\n"
        "there was rain below the ice associated with uncertain attenuation of the reflectivities\n"
        "in the ice.\n"
    )
    return attributes


COMMENTS = {
    "ier_error": (
        "Error in effective radius of ice particles due to error propagation,\n"
        "of ier = 3/(2 rho_i) IWC / alpha, using error for IWC and alpha as given in Hogan 2006."
    ),
    "ier_retrieval_status": (
        "This variable describes whether a retrieval was performed\n"
        "for each pixel, and its associated quality."
    ),
    "ier_inc_rain": (
        "This variable is the same as ier but it also contains ier values\n"
        "above rain. The ier values above rain have been severely affected\n"
        "by attenuation and should be used when the effect of attenuation\n"
        "is being studied."
    ),
}

DEFINITIONS = {
    "ier_retrieval_status": (
        "\n"
        "Value 0: No ice present.\n"
        "Value 1: Reliable retrieval.\n"
        "Value 2: Unreliable retrieval due to uncorrected attenuation from liquid\n"
        "         water below the ice (no liquid water path measurement available).\n"
        "Value 3: Retrieval performed but radar corrected for liquid attenuation\n"
        "         using radiometer liquid water path which is not always accurate.\n"
        "Value 4: Ice detected only by the lidar.\n"
        "Value 5: Ice detected by radar but rain below so no retrieval performed\n"
        "         due to very uncertain attenuation.\n"
        "Value 6: Clear sky above rain wet-bulb temperature less than 0degC: if\n"
        "         rain attenuation were strong then ice could be present but undetected.\n"
        "Value 7: Drizzle or rain that would have been classified as ice if the wet-bulb\n"
        "         temperature were less than 0degC: may be ice if temperature is in error."
    ),
}

IER_ATTRIBUTES = {
    "ier": MetaData(
        long_name="Ice effective radius",
        units="m-6",
        ancillary_variables="ier_error",
    ),
    "ier_inc_rain": MetaData(
        long_name="Ice effective radius including rain",
        units="m-6",
        comment=COMMENTS["ier_inc_rain"],
    ),
    "ier_error": MetaData(
        long_name="Random error in ice effective radius",
        units="m-6",
        comment=COMMENTS["ier_error"],
    ),
    "ier_retrieval_status": MetaData(
        long_name="Ice effective radius retrieval status",
        comment=COMMENTS["ier_retrieval_status"],
        definition=DEFINITIONS["ier_retrieval_status"],
        units="1",
    ),
}
