"""Module for creating Cloudnet liquid water content file using scaled-adiabatic
method.
"""

import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.categorize import atmos_utils
from cloudnetpy.datasource import DataSource
from cloudnetpy.exceptions import InvalidSourceFileError
from cloudnetpy.metadata import MetaData
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy.products.product_tools import CategorizeBits, get_is_rain


def generate_lwc(
    categorize_file: str,
    output_file: str,
    uuid: str | None = None,
) -> str:
    """Generates Cloudnet liquid water content product.

    This function calculates cloud liquid water content using the so-called
    adiabatic-scaled method. In this method, liquid water content measured by
    microwave radiometer is used to constrain the theoretical liquid water
    content of observed liquid clouds. The results are written in a netCDF file.

    Args:
        categorize_file: Categorize file name.
        output_file: Output file name.
        uuid: Set specific UUID for the file.

    Returns:
        str: UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_lwc
        >>> generate_lwc('categorize.nc', 'lwc.nc')

    References:
        Illingworth, A.J., R.J. Hogan, E. O'Connor, D. Bouniol, M.E. Brooks,
        J. Delanoé, D.P. Donovan, J.D. Eastment, N. Gaussiat, J.W. Goddard,
        M. Haeffelin, H.K. Baltink, O.A. Krasnov, J. Pelon, J. Piriou, A. Protat,
        H.W. Russchenberg, A. Seifert, A.M. Tompkins, G. van Zadelhoff, F. Vinit,
        U. Willén, D.R. Wilson, and C.L. Wrench, 2007: Cloudnet.
        Bull. Amer. Meteor. Soc., 88, 883–898, https://doi.org/10.1175/BAMS-88-6-883

    """
    with LwcSource(categorize_file) as lwc_source:
        lwc = Lwc(lwc_source)
        clouds = CloudAdjustor(lwc_source, lwc)
        lwc_error = LwcError(lwc_source, lwc)
        lwc_source.append_results(lwc.lwc, clouds.status, lwc_error.error)
        date = lwc_source.get_date()
        attributes = output.add_time_attribute(LWC_ATTRIBUTES, date)
        output.update_attributes(lwc_source.data, attributes)
        return output.save_product_file(
            "lwc",
            lwc_source,
            output_file,
            uuid,
            copy_from_cat=(
                "lwp",
                "lwp_error",
            ),
        )


class LwcSource(DataSource):
    """Data container for liquid water content calculations. Child of DataSource.

    This class reads input data from a categorize file and provides data
    structures and methods for holding the results.

    Args:
        categorize_file: Categorize file name.

    Attributes:
        lwp (ndarray): 1D liquid water path.
        lwp_error (ndarray): 1D error of liquid water path.
        is_rain (ndarray): 1D array denoting presence of rain.
        path_lengths (ndarray): 1D array of path lengths.
        atmosphere (dict): Dictionary containing interpolated fields `temperature`
            and `pressure`.
        categorize_bits (CategorizeBits): The :class:`CategorizeBits` instance.

    """

    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        if "lwp" not in self.dataset.variables:
            msg = "Liquid water path missing from the categorize file."
            raise InvalidSourceFileError(msg)
        self.lwp = self.getvar("lwp")
        self.lwp[self.lwp < 0] = 0
        self.lwp_error = self.getvar("lwp_error")
        self.is_rain = get_is_rain(categorize_file)
        self.path_lengths = utils.path_lengths_from_ground(self.getvar("height"))
        self.atmosphere = self._get_atmosphere(categorize_file)
        self.categorize_bits = CategorizeBits(categorize_file)

    def append_results(
        self,
        lwc: np.ndarray,
        status: np.ndarray,
        error: np.ndarray,
    ) -> None:
        self.append_data(lwc, "lwc", units="kg m-3")
        self.append_data(status, "lwc_retrieval_status")
        self.append_data(error, "lwc_error", units="dB")

    @staticmethod
    def _get_atmosphere(categorize_file: str) -> tuple[np.ndarray, np.ndarray]:
        fields = ["temperature", "pressure"]
        atmosphere = p_tools.interpolate_model(categorize_file, fields)
        return atmosphere["temperature"], atmosphere["pressure"]


class Lwc:
    """Class handling the actual LWC calculations.

    Args:
        lwc_source: The :class:`LwcSource` instance.

    Attributes:
        lwc_source (LwcSource): The :class:`LwcSource` instance.
        is_liquid (ndarray): 2D array denoting liquid.
        lwc_adiabatic (ndarray): 2D array storing adiabatic lwc.
        lwc (ndarray): 2D array of liquid water content (scaled with lwp).

    """

    def __init__(self, lwc_source: LwcSource):
        self.lwc_source = lwc_source
        self.height = lwc_source.getvar("height")
        self.is_liquid = self._get_liquid()
        self.lwc_adiabatic = self._init_lwc_adiabatic()
        self.lwc = self._adiabatic_lwc_to_lwc()
        self._mask_rain()

    def _get_liquid(self) -> np.ndarray:
        category_bits = self.lwc_source.categorize_bits.category_bits
        return category_bits.droplet

    def _init_lwc_adiabatic(self) -> np.ndarray:
        """Returns theoretical adiabatic lwc in liquid clouds (kg/m3)."""
        lwc_dz = atmos_utils.fill_clouds_with_lwc_dz(
            *self.lwc_source.atmosphere,
            self.is_liquid,
        )
        return atmos_utils.calc_adiabatic_lwc(lwc_dz, self.height)

    def _adiabatic_lwc_to_lwc(self) -> np.ndarray:
        """Initialises liquid water content (kg/m3).

        Calculates LWC for ALL profiles (rain, lwp > theoretical, etc.),
        """
        return atmos_utils.normalize_lwc_by_lwp(
            self.lwc_adiabatic,
            self.lwc_source.lwp,
            self.height,
        )

    def _mask_rain(self) -> None:
        is_rain = self.lwc_source.is_rain.astype(bool)
        self.lwc[is_rain, :] = ma.masked


class CloudAdjustor:
    """Adjusts clouds (where possible) so that theoretical and measured LWP agree.

    Args:
        lwc_source: The :class:`LwcSource` instance.
        lwc:  The :class:`Lwc` instance.

    Attributes:
        lwc_source (LwcSource): The :class:`LwcSource` instance.
        lwc (ndarray): Liquid water content data.
        is_liquid (ndarray): 2D array denoting liquid.
        lwc_adiabatic (ndarray): 2D array storing adiabatic lwc.
        echo (dict): Dictionary storing radar and lidar echos
        status (ndarray): 2D array storing lwc status classification

    """

    def __init__(self, lwc_source: LwcSource, lwc: Lwc):
        self.lwc_source = lwc_source
        self.lwc = lwc.lwc
        self.is_liquid = lwc.is_liquid
        self.lwc_adiabatic = lwc.lwc_adiabatic
        self.echo = self._get_echo()
        self.status = self._init_status()
        self._adjust_cloud_tops(self._find_adjustable_clouds())
        self._mask_rain()
        self._mask_missing()

    def _get_echo(self) -> dict:
        quality_bits = self.lwc_source.categorize_bits.quality_bits
        return {"radar": quality_bits.radar, "lidar": quality_bits.lidar}

    def _init_status(self) -> ma.MaskedArray:
        status = ma.zeros(self.is_liquid.shape, dtype=int)
        status[self.is_liquid] = 1
        return status

    def _adjust_cloud_tops(self, adjustable_clouds: np.ndarray) -> None:
        """Adjusts cloud top index so that measured lwc corresponds to theoretical
        value.
        """
        for time_index in np.unique(np.where(adjustable_clouds)[0]):
            base_index = np.where(adjustable_clouds[time_index, :])[0][0]
            self._update_status(time_index)
            self._adjust_lwc(time_index, base_index)

    def _update_status(self, time_ind: np.ndarray) -> None:
        alt_indices = np.where(self.is_liquid[time_ind, :])[0]
        self.status[time_ind, alt_indices] = 2

    def _adjust_lwc(self, time_ind: int, base_ind: int) -> None:
        lwc_base = self.lwc_adiabatic[time_ind, base_ind]
        distance_from_base = 1
        while True:
            top_ind = base_ind + distance_from_base
            lwc_top = lwc_base * (distance_from_base + 1)
            self.lwc_adiabatic[time_ind, top_ind] = lwc_top
            if not self.status[time_ind, top_ind]:
                self.status[time_ind, top_ind] = 3
            if self._has_converged(time_ind) or self._out_of_bound(top_ind):
                break
            distance_from_base += 1

    def _has_converged(self, ind: int) -> bool:
        lwc_sum = ma.sum(self.lwc_adiabatic[ind, :] * self.lwc_source.path_lengths)
        return lwc_sum > self.lwc_source.lwp[ind]

    def _out_of_bound(self, ind: int) -> bool:
        return ind >= self.lwc.shape[1] - 1

    def _find_adjustable_clouds(self) -> np.ndarray:
        top_clouds = self._find_topmost_clouds()
        detection_type = self._find_echo_combinations_in_liquid()
        detection_type[~top_clouds] = 0
        lidar_only_clouds = self._find_lidar_only_clouds(detection_type)
        top_clouds[~lidar_only_clouds, :] = 0
        return self._remove_good_profiles(top_clouds)

    def _find_topmost_clouds(self) -> np.ndarray:
        top_clouds = np.copy(self.is_liquid)
        cloud_edges = top_clouds[:, :-1][:, ::-1] < top_clouds[:, 1:][:, ::-1]
        topmost_bases = self.is_liquid.shape[1] - 1 - np.argmax(cloud_edges, axis=1)
        for n, base in enumerate(topmost_bases):
            top_clouds[n, :base] = 0
        return top_clouds

    def _find_echo_combinations_in_liquid(self) -> np.ndarray:
        """Classifies liquid clouds by detection type: 1=lidar, 2=radar, 3=both."""
        lidar_detected = (self.is_liquid & self.echo["lidar"]).astype(int)
        radar_detected = (self.is_liquid & self.echo["radar"]).astype(int) * 2
        return lidar_detected + radar_detected

    @staticmethod
    def _find_lidar_only_clouds(detection: np.ndarray) -> np.ndarray:
        """Finds top clouds that contain only lidar-detected pixels.

        Args:
            detection: Array of integers where 1=lidar, 2=radar, 3=both.

        Returns:
            Boolean array containing top-clouds that are detected only by lidar.

        """
        sum_of_cloud_pixels = ma.sum(detection > 0, axis=1)
        sum_of_detection_type = ma.sum(detection, axis=1)
        return sum_of_cloud_pixels / sum_of_detection_type == 1

    def _remove_good_profiles(self, top_clouds: np.ndarray) -> np.ndarray:
        no_rain = ~self.lwc_source.is_rain.astype(bool)
        lwp_difference = self._find_lwp_difference()
        dubious_profiles = (lwp_difference < 0) & no_rain
        top_clouds[~dubious_profiles, :] = 0
        return top_clouds

    def _find_lwp_difference(self) -> np.ndarray:
        """Returns difference of theoretical LWP and measured LWP (g/m2).

        In theory, this difference should be always positive. Negative values
        indicate missing (or too narrow) liquid clouds.
        """
        lwc_sum = ma.sum(self.lwc_adiabatic * self.lwc_source.path_lengths, axis=1)
        return lwc_sum - self.lwc_source.lwp

    def _mask_rain(self) -> None:
        is_rain = self.lwc_source.is_rain.astype(bool)
        self.status[is_rain, :] = 4

    def _mask_missing(self) -> None:
        is_missing = np.where(self.lwc_source.lwp == ma.masked)
        self.status[is_missing, :] = 4


class LwcError:
    """Calculates liquid water content error.

    Args:
        lwc_source: The :class:`LwcSource` instance.
        lwc: The :class:`Lwc` instance.

    Attributes:
        lwc_source (LwcSource): The :class:`LwcSource` instance.
        lwc (ndarray): Liquid water content data.
        error (ndarray): 2D array storing lwc_error.

    """

    def __init__(self, lwc_source: LwcSource, lwc: Lwc):
        self.lwc = lwc.lwc
        self.lwc_source = lwc_source
        self.error = self._calculate_lwc_error()
        self._mask_rain()

    def _calculate_lwc_error(self) -> np.ndarray:
        lwc_relative_error = self._calc_lwc_relative_error()
        lwp_relative_error = self._calc_lwp_relative_error()
        combined_error = self._calc_combined_error(
            lwc_relative_error,
            lwp_relative_error,
        )
        return self._fill_error_array(combined_error)

    def _calc_lwc_relative_error(self) -> np.ndarray:
        lwc_gradient = self._calc_lwc_gradient()
        error = lwc_gradient / self.lwc / 2
        return self._limit_error(error, 5)

    def _calc_lwc_gradient(self) -> np.ndarray:
        if not isinstance(self.lwc, ma.MaskedArray):
            self.lwc = ma.masked_array(self.lwc)
        gradient_elements = np.gradient(self.lwc.filled(0))
        return utils.l2norm(*gradient_elements)

    def _calc_lwp_relative_error(self) -> np.ndarray:
        err = self.lwc_source.lwp_error
        value = self.lwc_source.lwp
        error = np.divide(err, value, out=np.zeros_like(err), where=value != 0)
        return self._limit_error(error, 10)

    @staticmethod
    def _limit_error(error: np.ndarray, max_value: float) -> np.ndarray:
        error[error > max_value] = max_value
        return error

    @staticmethod
    def _calc_combined_error(error_2d: np.ndarray, error_1d: np.ndarray) -> np.ndarray:
        error_1d_transposed = utils.transpose(error_1d)
        return utils.l2norm(error_2d, error_1d_transposed)

    def _fill_error_array(self, error_in: np.ndarray) -> ma.MaskedArray:
        lwc_error = ma.masked_all(self.lwc.shape)
        ind = ma.where(self.lwc)
        lwc_error[ind] = error_in[ind]
        return lwc_error

    def _mask_rain(self) -> None:
        is_rain = self.lwc_source.is_rain.astype(bool)
        self.error[is_rain, :] = ma.masked


COMMENTS = {
    "lwc": (
        "This variable was calculated for the profiles where the categorization data\n"
        "has diagnosed that liquid water is present and liquid water path is\n"
        "available from a coincident microwave radiometer. The model temperature and\n"
        "pressure were used to estimate the theoretical adiabatic liquid water\n"
        "content gradient for each cloud base and the adiabatic liquid water\n"
        "content is then scaled that its integral matches the radiometer measurement\n"
        "so that the liquid water content now follows a quasi-adiabatic profile.\n"
        "If the liquid layer is detected by the lidar only, there is the potential\n"
        "for cloud top height to be underestimated and so if the adiabatic integrated\n"
        "liquid water content is less than that measured by the microwave radiometer,\n"
        "the cloud top is extended until the adiabatic integrated liquid water\n"
        "content agrees with the value measured by the microwave radiometer.\n"
        "Missing values indicate that either\n"
        "1) a liquid water layer was diagnosed but no microwave radiometer data was\n"
        "   available,\n"
        "2) a liquid water layer was diagnosed but the microwave radiometer data was\n"
        "   unreliable; this may be because a melting layer was present in the\n"
        "   profile, or because the retrieved lwp was unphysical (values of zero\n"
        "   are not uncommon for thin supercooled liquid layers)\n"
        "3) that rain is present in the profile and therefore, the vertical extent of\n"
        "   liquid layers is difficult to ascertain."
    ),
    "lwc_error": (
        "This variable is an estimate of the random error in liquid water content\n"
        "due to the uncertainty in the microwave radiometer liquid water path\n"
        "retrieval and the uncertainty in cloud base and/or cloud top height."
    ),
    "lwc_retrieval_status": (
        "This variable describes whether a retrieval was performed for each pixel,\n"
        "and its associated quality, in the form of 6 different classes. The classes\n"
        "are defined in the definition attribute. The most reliable retrieval is that\n"
        "when both radar and lidar detect the liquid layer, and microwave radiometer\n"
        "data is present, indicated by the value 1. The next most reliable is when\n"
        "microwave radiometer data is used to adjust the cloud depth when the radar\n"
        "does not detect the liquid layer, indicated by the value 2, with a value of\n"
        "3 indicating the cloud pixels that have been added at cloud top to avoid\n"
        "the profile becoming superadiabatic. A value of 4 indicates that microwave\n"
        "radiometer data were not available or not reliable (melting level present\n"
        "or unphysical values) but the liquid layers were well defined. If cloud top\n"
        "was not well defined then this is indicated by a value of 5. The full\n"
        "retrieval of liquid water content, which requires reliable liquid water\n"
        "path from the microwave radiometer, was only performed for classes 1-3.\n"
        "No attempt is made to retrieve liquid water content when rain is present;\n"
        "this is indicated by the value 6."
    ),
}

DEFINITIONS = {
    "lwc_retrieval_status": utils.status_field_definition(
        {
            0: """No liquid water detected.""",
            1: """Reliable retrieval.""",
            2: """Adiabatic retrieval where cloud top has been adjusted to match
                  liquid water path from microwave radiometer because layer is
                  not detected by radar.""",
            3: """Adiabatic retrieval: new cloud pixels where cloud top has been
                  adjusted to match liquid water path from microwave radiometer
                  because layer is not detected by radar.""",
            4: """No retrieval: either no liquid water path is available or
                  liquid water path is uncertain.""",
            5: """No retrieval: liquid water layer detected only by the lidar
                  and liquid water path is unavailable or uncertain: cloud top
                  may be higher than diagnosed cloud top since lidar signal has
                  been attenuated.""",
            6: """Rain present: cloud extent is difficult to ascertain and
                  liquid water path also uncertain.""",
        }
    ),
}


LWC_ATTRIBUTES = {
    "lwc": MetaData(
        long_name="Liquid water content",
        comment=COMMENTS["lwc"],
        ancillary_variables="lwc_error",
        standard_name="mass_concentration_of_liquid_water_in_air",
    ),
    "lwc_error": MetaData(
        long_name="Random error in liquid water content, one standard deviation",
        comment=COMMENTS["lwc_error"],
        units="dB",
    ),
    "lwc_retrieval_status": MetaData(
        long_name="Liquid water content retrieval status",
        comment=COMMENTS["lwc_retrieval_status"],
        definition=DEFINITIONS["lwc_retrieval_status"],
        units="1",
    ),
}
