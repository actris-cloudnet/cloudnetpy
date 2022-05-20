"""Module for creating Cloudnet ice particle effective radius file using Z-T method."""
from collections import namedtuple
from typing import Optional

import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.categorize import atmos
from cloudnetpy.datasource import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products import product_tools
from cloudnetpy.products.product_tools import ProductClassification
from cloudnetpy.products.iwc import IwcSource

G_TO_KG = 0.001
rho_ice = 917  # kg m-3

Coefficients = namedtuple("Coefficients", "K2liquid0 ZT T Z c")


def generate_ier(categorize_file: str, output_file: str, uuid: Optional[str] = None) -> str:
    """Generates Cloudnet ice particle effective radius product.

    This function calculates ice particle effective radius using the Grieche et al. 2020 method
    which uses Hogan et al. 2006 to estimate ice water content and alpha from Delanoë et al. 2007.
    In this method, ice particle effective radius is calculated from attenuated-corrected
    radar reflectivity and model temperature. The results are written in a
    netCDF file.

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

    ier_source = IerSource(categorize_file)
    ice_classification = IceClassification(categorize_file)

    ier_source.append_ier_including_rain(ice_classification)
    ier_source.append_ier(ice_classification)
    ier_source.append_status(ice_classification)
    ier_source.append_ier_error(ice_classification)

    date = ier_source.get_date()
    attributes = output.add_time_attribute(IWC_ATTRIBUTES, date)
    attributes = _add_ier_comment(attributes, ier_source)
    output.update_attributes(ier_source.data, attributes)
    uuid = output.save_product_file("ier", ier_source, output_file, uuid)
    ier_source.close()
    return uuid


class IceClassification(ProductClassification):
    """Class storing the information about different ice types.
    Child of ProductClassification().
    """

    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.is_ice = self._find_ice()
        self.would_be_ice = self._find_would_be_ice()
        self.corrected_ice = self._find_corrected_ice()
        self.uncorrected_ice = self._find_uncorrected_ice()
        self.ice_above_rain = self._find_ice_above_rain()
        self.cold_above_rain = self._find_cold_above_rain()

    def _find_ice(self) -> np.ndarray:
        return (
            self.category_bits["falling"]
            & self.category_bits["cold"]
            & ~self.category_bits["melting"]
            & ~self.category_bits["insect"]
        )

    def _find_would_be_ice(self) -> np.ndarray:
        warm_falling = (
            self.category_bits["falling"]
            & ~self.category_bits["cold"]
            & ~self.category_bits["insect"]
        )
        return warm_falling | self.category_bits["melting"]

    def _find_corrected_ice(self) -> np.ndarray:
        return self.is_ice & self.quality_bits["attenuated"] & self.quality_bits["corrected"]

    def _find_uncorrected_ice(self) -> np.ndarray:
        return self.is_ice & self.quality_bits["attenuated"] & ~self.quality_bits["corrected"]

    def _find_ice_above_rain(self) -> np.ndarray:
        is_rain = utils.transpose(self.is_rain)
        return (self.is_ice * is_rain) == 1

    def _find_cold_above_rain(self) -> np.ndarray:
        is_cold = self.category_bits["cold"]
        is_rain = utils.transpose(self.is_rain)
        is_cold_rain = (is_cold * is_rain) == 1
        return is_cold_rain & ~self.category_bits["melting"]


class IerSource(DataSource):

    """Data container for ice effective radius calculations."""

    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.wl_band = utils.get_wl_band(float(self.getvar("radar_frequency")))
        self.coeffs = self._get_ier_coeffs()
        self.z_factor = self._get_z_factor()
        self.temperature = self._get_temperature(categorize_file)


    def append_ier_including_rain(self, ice_classification: IceClassification, iwc_source: Optional[IwcSource] = None) -> None:
        """Calculates ice effective radius (including ice above rain)."""
#        if iwc_source is None:
        ier_including_rain = self.ZT_to_ier("Z")
#        else:
#            ier_including_rain = self.iwc_to_ier(iwc_source)

        ier_including_rain[~ice_classification.is_ice] = ma.masked
        self.append_data(ier_including_rain, "ier_inc_rain")

    def append_ier(self, ice_classification: IceClassification) -> None:
        """Calculates ice effective radius"""
        ier = ma.copy(self.data["ier_inc_rain"][:])
        ier[ice_classification.ice_above_rain] = ma.masked

        self.append_data(ier, "ier")

    def append_ier_error(self, ice_classification: IceClassification) -> None:
        ier_error = ma.copy(self.data["ier_inc_rain"][:])
        ier_error[ice_classification.ice_above_rain] = ma.masked
        ier_error = ier_error * np.sqrt(0.4 ** 2 + 0.4 ** 2)

        self.append_data(ier_error, "ier_error")


    def append_status(self, ice_classification: IceClassification) -> None:
        """Returns information about the status of ier retrieval."""

        ier = self.data["ier"][:]
        retrieval_status = np.zeros(ier.shape, dtype=int)
        is_ier = ~ier.mask
        retrieval_status[is_ier] = 1
        retrieval_status[is_ier & ice_classification.uncorrected_ice] = 2
        retrieval_status[is_ier & ice_classification.corrected_ice] = 3
        retrieval_status[~is_ier & ice_classification.is_ice] = 4
        retrieval_status[ice_classification.ice_above_rain] = 5
        retrieval_status[ice_classification.cold_above_rain] = 6
        retrieval_status[ice_classification.would_be_ice & (retrieval_status == 0)] = 7
        self.append_data(retrieval_status, "ier_retrieval_status")

    def _get_ier_coeffs(self) -> Coefficients:
        """Returns coefficients for ice particle effective radius retrieval.

        References:
            Hogan et.al. 2006, https://doi.org/10.1175/JAM2340.1
        """
        if self.wl_band == 0:
            return Coefficients(0.878, -0.000205, -0.0015, 0.0016, -1.52)
        return Coefficients(0.669, -0.000296, -0.00193, -0.000, -1.502)

    def _get_z_factor(self) -> float:
        """Returns empirical scaling factor for radar echo."""
        return float(utils.lin2db(self.coeffs.K2liquid0 / 0.93))

    @staticmethod
    def _get_temperature(categorize_file: str) -> np.ndarray:
        """Returns interpolated temperatures in Celsius."""
        atmosphere = product_tools.interpolate_model(categorize_file, "temperature")
        return atmos.k2c(atmosphere["temperature"])

    def ZT_to_ier(self, z_variable: str) -> np.ndarray:
        """Calculates temperature weighted z, i.e. ice particle effective radius (mu m-3).
        """
        if z_variable == "Z":
            temperature = self.temperature
        else:
            temperature = ma.mean(self.temperature, axis=0)
        z_scaled = self.getvar(z_variable) + self.z_factor
        coeffs = self.coeffs
        return (
            10
            ** (
                coeffs.ZT * z_scaled * temperature
                + coeffs.T * temperature
                + coeffs.Z * z_scaled
                + coeffs.c
            )
            * 1.0e6 * 3/(2*rho_ice)
        )

def _add_ier_comment(attributes: dict, ier: IerSource) -> dict:
    freq = utils.get_frequency(ier.wl_band)
    coeffs = ier.coeffs
    factor = round((coeffs[0] / 0.93) * 1000) / 1000
    attributes["ier"] = attributes["ier"]._replace(
        comment=f"This variable was calculated from the {freq}-GHz radar reflectivity \n"
        f"factor after correction for gaseous attenuation, and temperature taken from \n"
        f"a forecast model, using the following empirical formula: \n"
        rf"log10($\alpha$[g m-3]) = freq{coeffs[1]}Z[dBZ]T[degC] + "
                f"{coeffs[3]}Z[dBZ] + {coeffs[2]}T[degC] + {coeffs[4]}.\n"
        "In this formula $\\mathregular{\\alpha}$ is taken to be defined such that all \n"
        "frequencies of radar would measure the same Z in Rayleigh scattering ice.\n"
        "However, the radar is more likely to have been calibrated such that all \n"
        "frequencies would measure the same Z in Rayleigh scattering\n"
        "liquid cloud at 0 degrees C. The measured Z is therefore multiplied by \n"
        f"|K(liquid,0degC,{freq}GHz)|^2/0.93 = {factor}  before applying this formula.\n"
        "The formula has been used where the \"categorization\" data has diagnosed \n"
        "that the radar echo is due to ice, but note that in some cases\n"
        "supercooled drizzle will erroneously be identified as ice. Missing data \n"
        "indicates either that ice cloud was present but it was only\n"
        "detected by the lidar so its ice water content could not be estimated, or \n"
        "that there was rain below the ice associated with uncertain\n"
        "attenuation of the reflectivities in the ice.\n"
        "Note that where microwave radiometer liquid water path was available it was \n"
        "used to correct the radar for liquid attenuation when liquid\n"
        "cloud occurred below the ice; this is indicated a value of 3 in the \n"
        "iwc_retrieval_status variable.  There is some uncertainty in this\n"
        "prodedure which is reflected by an increase in the associated values in the \n"
        "iwc_error variable. When microwave radiometer data were not available and \n"
        "liquid cloud occurred below the ice, the retrieval was still performed but its\n"
        "reliability is questionable due to the uncorrected liquid water attenuation.\n"
        "This is indicated by a value of 2 in the iwc_retrieval_status\n"
        "variable, and an increase in the value of the iwc_error variable"
    )
    return attributes


COMMENTS = {
    "ier": (
        "Ice effective radius Hogen et al. 2006."
    ),
    "ier_retrieval_status_comment": (
        "This variable describes whether a retrieval was performed for each\n"
        "pixel, and its associated quality, in the form of 8 different classes.\n"
        "The classes are defined in the definition and long_definition attributes. \n"
        "The most reliable retrieval is that without any rain or liquid\n"
        "cloud beneath, indicated by the value 1, then the next most reliable \n"
        "is when liquid water attenuation has been corrected using a microwave\n"
        "radiometer, indicated by the value 3, while a value 2 indicates that \n"
        "liquid water cloud was present but microwave radiometer data were not\n"
        "available so no correction was performed. No attempt is made to retrieve \n"
        "ice water content when rain is present below the ice; this is\n"
        "indicated by the value 5."
    ),
    "ier_error": (
        "Error in effective radius of ice particles due to error propagation, \n"
        "of ier = 3/(2 rho_i) IWC / alpha, using error for IWC and alpha as given in Hogan 2006."
    ),
    "ier_retrieval_status": (
        "This variable describes whether a retrieval was performed\n"
        "for each pixel, and its associated quality."
    ),
    "ier_inc_rain": (
        "This variable is the same as iwc, except that values of iwc in \n"
        "ice above rain have been included. This variable contains values \n"
        "which have been severely affected by attenuation and should only \n"
        "be used when the effect of attenuation is being studied"
    ),
}

DEFINITIONS = {
    "ier_retrieval_status": (
        "\n"
        "Value 0: No ice\n"
        "Value 1: Reliable retrieval\n"
        "Value 2: Unreliable: uncorrected attenuation\n"
        "Value 3: Retrieval with correction for liquid atten.\n"
        "Value 4: Ice detected only by the lidar\n"
        "Value 5: Ice above rain: no retrieval\n"
        "Value 6: Clear sky above rain\n"
        "Value 7: Would be identified as ice if below freezing"
    ),
    "ier_retrieval_status_long": (
        "\n"
        "Value 0: No ice present\n"
        "Value 1: Reliable retrieval\n"
        "Value 2: Unreliable retrieval due to uncorrected attenuation from liquid \n"
        "         water below the ice (no liquid water path measurement available)\n"
        "Value 3: Retrieval performed but radar corrected for liquid attenuation \n"
        "         using radiometer liquid water path which is not always accurate\n"
        "Value 4: Ice detected only by the lidar\n"
        "Value 5: Ice detected by radar but rain below so no retrieval performed \n"
        "         due to very uncertain attenuation\n"
        "Value 6: Clear sky above rain wet-bulb temperature less than 0degC: if \n"
        "         rain attenuation were strong then ice could be present but undetected\n"
        "Value 7: Drizzle or rain that would have been classified as ice if the \n"
        "         wet-bulb temperature were less than 0degC: may be ice if \n"
        "         temperature is in error"
    )
}

IWC_ATTRIBUTES = {
    "ier": MetaData(
        long_name="Effective radius ice particles",
        units=r"$\mu$\,m",
        comment=COMMENTS["ier"],
        ancillary_variables="ier_error ier_sensitivity ier_bias",
    ),
    "ier_error": MetaData(
        long_name="Error in ice particle effective radius",
        units=r"$\mu$\,m",
        comment=COMMENTS["ier_error"],
    ),
    "ier_retrieval_status": MetaData(
        long_name="Effective radius ice particles retrieval status",
        comment=COMMENTS["ier_retrieval_status"],
        definition=DEFINITIONS["ier_retrieval_status"],
        units="",
    ),
    "ier_inc_rain": MetaData(
        long_name="Effective radius ice particles",
        units=r"$\mu$\,m",
        comment=COMMENTS["ier_inc_rain"],
        ancillary_variables="ier_sensitivity ier_bias ier_error",
    ),
}
