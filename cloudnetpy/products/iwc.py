"""Module for creating Cloudnet ice water content file using Z-T method."""
import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.constants import G_TO_KG
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import IceClassification, IceSource


def generate_iwc(
    categorize_file: str,
    output_file: str,
    uuid: str | None = None,
) -> str:
    """Generates Cloudnet ice water content product.

    This function calculates ice water content using the so-called Z-T method.
    In this method, ice water content is calculated from attenuated-corrected
    radar reflectivity and model temperature. The results are written in a
    netCDF file.

    Args:
    ----
        categorize_file: Categorize file name.
        output_file: Output file name.
        uuid: Set specific UUID for the file.

    Returns:
    -------
        UUID of the generated file.

    Examples:
    --------
        >>> from cloudnetpy.products import generate_iwc
        >>> generate_iwc('categorize.nc', 'iwc.nc')

    References:
    ----------
        Hogan, R.J., M.P. Mittermaier, and A.J. Illingworth, 2006:
        The Retrieval of Ice Water Content from Radar Reflectivity Factor and
        Temperature and Its Use in Evaluating a Mesoscale Model.
        J. Appl. Meteor. Climatol., 45, 301–317, https://doi.org/10.1175/JAM2340.1

    """
    product = "iwc"
    with IwcSource(categorize_file, product) as iwc_source:
        ice_classification = IceClassification(categorize_file)
        iwc_source.append_main_variable_including_rain(ice_classification)
        iwc_source.append_main_variable(ice_classification)
        iwc_source.append_bias()
        iwc_source.append_sensitivity()
        lwp_prior, bias = iwc_source.append_error(ice_classification)
        iwc_source.append_status(ice_classification)
        date = iwc_source.get_date()
        attributes = output.add_time_attribute(IWC_ATTRIBUTES, date)
        attributes = _add_iwc_comment(attributes, iwc_source)
        attributes = _add_iwc_error_comment(attributes, lwp_prior, bias)
        output.update_attributes(iwc_source.data, attributes)
        return output.save_product_file(product, iwc_source, output_file, uuid)


class IwcSource(IceSource):
    """Data container for ice water content calculations."""

    def append_sensitivity(self) -> None:
        """Calculates iwc sensitivity."""
        sensitivity = self._convert_z("Z_sensitivity")
        self.append_data(sensitivity, f"{self.product}_sensitivity")

    def append_bias(self) -> None:
        """Calculates iwc bias."""
        bias = self.getvar("Z_bias") * self.coefficients.Z * 10
        self.append_data(bias, f"{self.product}_bias")

    def append_error(self, ice_classification: IceClassification) -> tuple:
        """Estimates error of ice water content."""

        def _calc_random_error() -> np.ndarray:
            scaled_temperature = self.coefficients.ZT * self.temperature
            scaled_temperature += self.coefficients.Z
            return self.getvar("Z_error") * scaled_temperature * 10

        def _calc_error_in_uncorrected_ice() -> float:
            spec_liq_atten = 1.0 if self.wl_band == 0 else 4.5
            liq_atten_scaled = spec_liq_atten * self.coefficients.Z
            return lwp_prior * G_TO_KG * liq_atten_scaled * 2 * 10

        lwp_prior = 250  # g m-2
        retrieval_uncertainty = 1.7  # dB
        random_error = _calc_random_error()
        error_uncorrected = _calc_error_in_uncorrected_ice()
        iwc_error = utils.l2norm(retrieval_uncertainty, random_error)
        iwc_error[ice_classification.uncorrected_ice] = utils.l2norm(
            retrieval_uncertainty,
            error_uncorrected,
        )
        iwc_error[
            (~ice_classification.is_ice | ice_classification.ice_above_rain)
        ] = ma.masked
        self.append_data(iwc_error, f"{self.product}_error")
        return lwp_prior, retrieval_uncertainty


def _add_iwc_error_comment(attributes: dict, lwp_prior, uncertainty: float) -> dict:
    attributes["iwc_error"] = attributes["iwc_error"]._replace(
        comment="This variable is an estimate of the one-standard-deviation random\n"
        "error in ice water content due to both the uncertainty of the retrieval\n"
        f"(about {uncertainty} dB), and the random error in radar reflectivity\n"
        "factor from which ice water content was calculated. When liquid water is\n"
        "present beneath the ice but no microwave radiometer data were available to\n"
        "correct for the associated attenuation, the error also includes a\n"
        f"contribution equivalent to approximately {lwp_prior} g m-2 of liquid water\n"
        "path being uncorrected for.",
    )
    return attributes


def _add_iwc_comment(attributes: dict, iwc: IwcSource) -> dict:
    freq = utils.get_frequency(iwc.wl_band)
    coeffs = iwc.coefficients
    factor = round((coeffs[0] / 0.93) * 1000) / 1000
    attributes["iwc"] = attributes["iwc"]._replace(
        comment=f"This variable was calculated from the {freq}-GHz radar reflectivity\n"
        "factor after correction for gaseous attenuation, and temperature taken from\n"
        f"a forecast model, using the following empirical formula:\n"
        f"log10(iwc[g m-3]) = {coeffs[1]}Z[dBZ]T[degC] + {coeffs[3]}Z[dBZ]\n"
        f"+ {coeffs[2]}T[degC] + {coeffs[4]}. In this formula Z is taken to be\n"
        "defined such that all frequencies of radar would measure the same Z in\n"
        "Rayleigh scattering ice. However, the radar is more likely to have been\n"
        "calibrated such that all frequencies would measure the same Z in Rayleigh\n"
        "scattering liquid cloud at 0 degrees C. The measured Z is therefore\n"
        f"multiplied by |K(liquid,0degC,{freq}GHz)|^2/0.93 = {factor} before applying\n"
        'this formula. The formula has been used where the "categorization" data has\n'
        "diagnosed that the radar echo is due to ice, but note that in some cases\n"
        "supercooled drizzle will erroneously be identified as ice. Missing data\n"
        "indicates either that ice cloud was present but it was only detected by\n"
        "the lidar so its ice water content could not be estimated, or that there\n"
        "was rain below the ice associated with uncertain attenuation of the\n"
        "reflectivities in the ice. Note that where microwave radiometer liquid water\n"
        "path was available it was used to correct the radar for liquid attenuation\n"
        "when liquid cloud occurred below the ice; this is indicated a value of 3\n"
        "in the iwc_retrieval_status variable. There is some uncertainty in this\n"
        "procedure which is reflected by an increase in the associated values\n"
        "in the iwc_error variable. When microwave radiometer data were not available\n"
        "and liquid cloud occurred below the ice, the retrieval was still performed\n"
        "but its reliability is questionable due to the uncorrected liquid water\n"
        "attenuation. This is indicated by a value of 2 in the iwc_retrieval_status\n"
        "variable, and an increase in the value of the iwc_error variable.",
    )
    return attributes


COMMENTS = {
    "iwc_bias": (
        "This variable is an estimate of the possible systematic error in \n"
        "ice water content due to the calibration error of the radar \n"
        "reflectivity factor from which it was calculated."
    ),
    "iwc_sensitivity": (
        "This variable is an estimate of the minimum detectable ice water\n"
        "content as a function of height."
    ),
    "iwc_retrieval_status": (
        "This variable describes whether a retrieval was performed\n"
        "for each pixel, and its associated quality."
    ),
    "iwc_inc_rain": (
        "This variable is the same as iwc but it also contains iwc values\n"
        "above rain. The iwc values above rain have been severely affected\n"
        "by attenuation and should be used when the effect of attenuation\n"
        "is being studied."
    ),
}

DEFINITIONS = {
    "iwc_retrieval_status": (
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
        "Value 6: Clear sky above rain and wet-bulb temperature less than 0degC:\n"
        "         if rain attenuation is strong, ice could be present but undetected.\n"
        "Value 7: Drizzle or rain that would have been classified as ice if the\n"
        "         wet-bulb temperature were less than 0degC: may be ice if\n"
        "         temperature is in error."
    ),
}

IWC_ATTRIBUTES = {
    "iwc": MetaData(
        long_name="Ice water content",
        units="kg m-3",
        ancillary_variables="iwc_error iwc_sensitivity iwc_bias",
    ),
    "iwc_inc_rain": MetaData(
        long_name="Ice water content including rain",
        units="kg m-3",
        comment=COMMENTS["iwc_inc_rain"],
    ),
    "iwc_error": MetaData(
        long_name="Random error in ice water content",
        units="dB",
    ),
    "iwc_bias": MetaData(
        long_name="Possible bias in ice water content",
        units="dB",
        comment=COMMENTS["iwc_bias"],
    ),
    "iwc_sensitivity": MetaData(
        long_name="Minimum detectable ice water content",
        units="kg m-3",
        comment=COMMENTS["iwc_sensitivity"],
    ),
    "iwc_retrieval_status": MetaData(
        long_name="Ice water content retrieval status",
        comment=COMMENTS["iwc_retrieval_status"],
        definition=DEFINITIONS["iwc_retrieval_status"],
        units="1",
    ),
}
