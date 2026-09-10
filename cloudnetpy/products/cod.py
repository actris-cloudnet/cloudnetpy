"""Module for creating Cloudnet cloud optical depth product."""

from os import PathLike
from uuid import UUID

import numpy as np
import numpy.typing as npt
from numpy import ma

from cloudnetpy import constants, output, utils
from cloudnetpy.datasource import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.der import DerSource
from cloudnetpy.products.ier import IerSource
from cloudnetpy.products.iwc import IwcSource
from cloudnetpy.products.lwc import CloudAdjustor, Lwc, LwcError, LwcSource
from cloudnetpy.products.product_tools import IceClassification

# Density of liquid water (kg m-3)
RHO_WATER = 1000

# Droplet effective radius (m) assumed when it cannot be retrieved from radar
DEFAULT_ASSUMED_DER = 10e-6

# Radar-retrieved droplet effective radius (m) outside this range is rejected
DER_VALID_RANGE = (2e-6, 50e-6)

# Measured LWP above this multiple of the adiabatic LWP of the detected liquid
# layers indicates liquid in layers not classified as droplets
LWP_RATIO_LIMIT = 2.0

# Relative error assumed for the droplet effective radius when not retrieved
ASSUMED_DER_REL_ERROR = 0.4


def generate_cod(
    categorize_file: str | PathLike,
    output_file: str | PathLike,
    uuid: str | UUID | None = None,
    assumed_der: float = DEFAULT_ASSUMED_DER,
) -> UUID:
    """Generates Cloudnet cloud optical depth product.

    This function calculates the visible extinction coefficient of liquid and
    ice clouds and integrates it over the profile to give cloud optical depth.
    Liquid extinction is derived from the adiabatic-scaled liquid water content
    (see :func:`generate_lwc`) and the LWP-scaled droplet effective radius
    (see :func:`generate_der`). Ice extinction is derived from ice water content
    and ice effective radius, both retrieved from radar reflectivity and model
    temperature (see :func:`generate_iwc` and :func:`generate_ier`). Extinction
    is calculated in the geometric optics limit, i.e. as 3/2 times the mass
    content divided by the particle density and effective radius. The results
    are written in a netCDF file.

    Args:
        categorize_file: Categorize file name.
        output_file: Output file name.
        uuid: Set specific UUID for the file.
        assumed_der: Droplet effective radius (m) used in liquid layers where
            the radar-based retrieval is not available.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_cod
        >>> generate_cod('categorize.nc', 'cod.nc')

    References:
        Frisch, S., Shupe, M., Djalalova, I., Feingold, G., & Poellot, M. (2002).
        The Retrieval of Stratus Cloud Droplet Effective Radius with Cloud Radars,
        Journal of Atmospheric and Oceanic Technology, 19(6), 835-842.
        https://doi.org/10.1175/1520-0426(2002)019%3C0835:TROSCD%3E2.0.CO;2

        Hogan, R. J., Mittermaier, M. P., & Illingworth, A. J. (2006). The
        Retrieval of Ice Water Content from Radar Reflectivity Factor and
        Temperature and Its Use in Evaluating a Mesoscale Model, Journal of
        Applied Meteorology and Climatology, 45(2), 301-317.
        https://doi.org/10.1175/JAM2340.1

        Delanoë, J., Protat, A., Bouniol, D., Heymsfield, A., Bansemer, A., &
        Brown, P. (2007). The Characterization of Ice Cloud Properties from
        Doppler Radar Measurements, Journal of Applied Meteorology and
        Climatology, 46(10), 1682-1698. https://doi.org/10.1175/JAM2543.1

        Griesche, H. J., Seifert, P., Ansmann, A., Baars, H., Barrientos
        Velasco, C., Bühl, J., Engelmann, R., Radenz, M., Zhenping, Y., &
        Macke, A. (2020): Application of the shipborne remote sensing supersite
        OCEANET for profiling of Arctic aerosols and clouds during Polarstern
        cruise PS106, Atmos. Meas. Tech., 13, 5335–5358.
        https://doi.org/10.5194/amt-13-5335-2020

    """
    uuid = utils.get_uuid(uuid)
    with OpticalDepthSource(categorize_file, assumed_der) as od_source:
        od_source.append_liquid_extinction()
        od_source.append_ice_extinction()
        od_source.append_extinction_status()
        od_source.append_optical_depths()
        od_source.append_optical_depth_error()
        od_source.append_optical_depth_status()
        date = od_source.get_date()
        attributes = output.add_time_attribute(dict(OPTICAL_DEPTH_ATTRIBUTES), date)
        attributes = _add_extinction_comments(attributes, od_source)
        output.update_attributes(od_source.data, attributes)
        output.save_product_file(
            "cod",
            od_source,
            output_file,
            uuid,
            copy_from_cat=("lwp", "lwp_error"),
        )
    return uuid


class OpticalDepthSource(DataSource):
    """Data container for cloud optical depth calculations."""

    def __init__(self, categorize_file: str | PathLike, assumed_der: float) -> None:
        if not np.isfinite(assumed_der) or assumed_der <= 0:
            msg = "Assumed droplet effective radius must be finite and positive."
            raise ValueError(msg)
        super().__init__(categorize_file)
        self.categorize_file = categorize_file
        self.assumed_der = assumed_der
        self.height_agl: npt.NDArray
        self.path_lengths = utils.path_lengths_from_ground(self.height_agl)
        self.ice_classification = IceClassification(categorize_file)
        self.is_rain = self.ice_classification.is_rain.astype(bool)
        self.is_liquid = self.ice_classification.category_bits.droplet
        self.has_lwp = "lwp" in self.dataset.variables
        self.is_der_assumed = np.zeros(self.is_liquid.shape, dtype=bool)
        self.is_lidar_only_ice = np.zeros(self.is_liquid.shape, dtype=bool)
        self.is_lwp_inconsistent = np.zeros(self.is_liquid.shape[0], dtype=bool)
        self._rel_error: dict[str, ma.MaskedArray] = {}

    def append_liquid_extinction(self) -> None:
        """Calculates liquid extinction from LWC and droplet effective radius."""
        lwc, lwc_rel_error, lwp_adiabatic = self._get_lwc()
        self.is_lwp_inconsistent = self._find_inconsistent_lwp(lwp_adiabatic)
        der, der_rel_error = self._get_der()
        der = ma.masked_outside(der, *DER_VALID_RANGE)
        self.is_der_assumed = ~ma.getmaskarray(lwc) & ma.getmaskarray(der)
        der_filled = ma.filled(der, self.assumed_der)
        der_rel_error_filled = ma.filled(der_rel_error, ASSUMED_DER_REL_ERROR).copy()
        der_rel_error_filled[self.is_der_assumed] = ASSUMED_DER_REL_ERROR
        # Includes pixels added at lidar-only cloud tops by the lwc retrieval
        extinction = 3 * lwc / (2 * RHO_WATER * der_filled)
        rel_error = utils.l2norm(ma.filled(lwc_rel_error, 0), der_rel_error_filled)
        self._rel_error["liquid"] = ma.masked_where(
            ma.getmaskarray(extinction), rel_error
        )
        self.append_data(extinction, "extinction_liquid")
        self.append_data(
            _relative_to_db(self._rel_error["liquid"]), "extinction_liquid_error"
        )

    def append_ice_extinction(self) -> None:
        """Calculates ice extinction from IWC and ice effective radius."""
        iwc, iwc_error = self._get_iwc()
        ier = self._get_ier()
        self.is_lidar_only_ice = self.ice_classification.is_ice & ma.getmaskarray(iwc)
        extinction = 3 * iwc / (2 * constants.RHO_ICE * ier)
        extinction[~self.ice_classification.is_ice] = ma.masked
        extinction[self.is_rain, :] = ma.masked
        rel_error = utils.db2lin(ma.array(iwc_error, copy=True)) - 1
        self._rel_error["ice"] = ma.masked_where(ma.getmaskarray(extinction), rel_error)
        error = ma.array(iwc_error, copy=True)
        error[ma.getmaskarray(extinction)] = ma.masked
        self.append_data(extinction, "extinction_ice")
        self.append_data(error, "extinction_ice_error")

    def append_optical_depth_error(self) -> None:
        """Estimates the error of the column optical depth.

        Errors are assumed fully correlated within a profile for each phase
        (they are dominated by LWP, effective radius and Z-T relation
        uncertainties) and independent between liquid and ice.
        """
        tau = self.data["optical_depth"][:]
        abs_errors = [
            self._integrate(ma.filled(self._rel_error[phase], 0) * ma.filled(ext, 0))
            for phase, ext in (
                ("liquid", self.data["extinction_liquid"][:]),
                ("ice", self.data["extinction_ice"][:]),
            )
        ]
        abs_error = utils.l2norm(*abs_errors)
        error = _relative_to_db(abs_error / ma.masked_less_equal(tau, 0))
        self.append_data(error, "optical_depth_error")

    def append_extinction_status(self) -> None:
        """Adds pixel-wise retrieval status."""
        is_liquid = ~ma.getmaskarray(self.data["extinction_liquid"][:])
        is_ice = ~ma.getmaskarray(self.data["extinction_ice"][:])
        status = np.zeros(is_liquid.shape, dtype=int)
        status[is_liquid] = 1
        status[is_liquid & self.is_der_assumed] = 2
        status[is_ice] = 3
        status[is_ice & self.ice_classification.corrected_ice] = 4
        status[is_liquid & is_ice] = 5
        status[self._find_cloud_without_retrieval()] = 6
        self.append_data(status, "extinction_retrieval_status")

    def append_optical_depths(self) -> None:
        """Integrates extinction over the profile."""
        tau_liquid = self._integrate(self.data["extinction_liquid"][:])
        tau_ice = self._integrate(self.data["extinction_ice"][:])
        tau = tau_liquid + tau_ice
        no_retrieval = self._find_profiles_without_retrieval()
        for array in (tau_liquid, tau_ice, tau):
            array[no_retrieval] = ma.masked
        self.append_data(tau_liquid, "optical_depth_liquid")
        self.append_data(tau_ice, "optical_depth_ice")
        self.append_data(tau, "optical_depth")

    def append_optical_depth_status(self) -> None:
        """Adds profile-wise retrieval status."""
        is_cloud = self.is_liquid | self.ice_classification.is_ice
        status = np.zeros(is_cloud.shape[0], dtype=int)
        status[np.any(is_cloud, axis=1)] = 1
        status[np.any(self.is_der_assumed, axis=1)] = 2
        status[np.any(self.is_lidar_only_ice, axis=1)] = 3
        status[self.is_lwp_inconsistent] = 4
        status[self._find_profiles_without_retrieval()] = 5
        self.append_data(status, "optical_depth_retrieval_status")

    def _integrate(self, extinction: npt.NDArray) -> ma.MaskedArray:
        return ma.sum(ma.filled(extinction, 0) * self.path_lengths, axis=1)

    def _find_cloud_without_retrieval(self) -> npt.NDArray:
        missing_ice = self.ice_classification.is_ice & self._is_missing(
            "extinction_ice"
        )
        return self._find_missing_liquid() | missing_ice

    def _find_missing_liquid(self) -> npt.NDArray:
        return self.is_liquid & self._is_missing("extinction_liquid")

    def _is_missing(self, key: str) -> npt.NDArray:
        return ma.getmaskarray(self.data[key][:])

    def _find_profiles_without_retrieval(self) -> npt.NDArray:
        missing_liquid = self._find_missing_liquid()
        uncorrected_ice = self.ice_classification.uncorrected_ice
        return (
            self.is_rain
            | np.any(missing_liquid, axis=1)
            | np.any(uncorrected_ice, axis=1)
        )

    def _find_inconsistent_lwp(self, lwp_adiabatic: npt.NDArray) -> npt.NDArray:
        """Finds profiles where LWP exceeds what the detected layers can hold."""
        if not self.has_lwp:
            return np.zeros(len(lwp_adiabatic), dtype=bool)
        lwp = ma.filled(self.getvar("lwp"), 0)
        lwp_adiabatic = ma.filled(lwp_adiabatic, 0)
        has_liquid = lwp_adiabatic > 0
        return has_liquid & (lwp > LWP_RATIO_LIMIT * lwp_adiabatic)

    def _get_lwc(self) -> tuple[ma.MaskedArray, npt.NDArray, npt.NDArray]:
        """Returns LWC (kg m-3), its relative error and adiabatic LWP (kg m-2).

        Without a microwave radiometer, LWC is masked everywhere and the
        liquid layers get no retrieval.
        """
        shape = self.is_liquid.shape
        if not self.has_lwp:
            return ma.masked_all(shape), np.zeros(shape), np.zeros(shape[0])
        with LwcSource(self.categorize_file) as lwc_source:
            lwc = Lwc(lwc_source)
            status = CloudAdjustor(lwc_source, lwc).status
            rel_error = LwcError(lwc_source, lwc).error
            lwp_positive = lwc_source.lwp > 0
        lwp_adiabatic = ma.sum(lwc.lwc_adiabatic * self.path_lengths, axis=1)
        valid = np.isin(status, (1, 2, 3)) & utils.transpose(lwp_positive)
        return ma.masked_where(~valid, lwc.lwc), rel_error, lwp_adiabatic

    def _get_der(self) -> tuple[ma.MaskedArray, ma.MaskedArray]:
        """Returns LWP-scaled droplet effective radius (m) and its relative error."""
        if not self.has_lwp:
            return ma.masked_all(self.is_liquid.shape), ma.masked_all(
                self.is_liquid.shape
            )
        with DerSource(self.categorize_file) as der_source:
            der_source.append_der()
            der = der_source.data["der_scaled"][:]
            error = der_source.data["der_scaled_error"][:]
        return der, error / der

    def _get_iwc(self) -> tuple[ma.MaskedArray, ma.MaskedArray]:
        """Returns IWC (kg m-3) and its random error (dB)."""
        with IwcSource(self.categorize_file, "iwc") as iwc_source:
            iwc_source.append_icy_data(self.ice_classification)
            iwc_source.append_error(self.ice_classification)
            return iwc_source.data["iwc"][:], iwc_source.data["iwc_error"][:]

    def _get_ier(self) -> ma.MaskedArray:
        with IerSource(self.categorize_file, "ier") as ier_source:
            ier_source.append_icy_data(self.ice_classification)
            ier_source.convert_units()
            return ier_source.data["ier"][:]


def _relative_to_db(rel_error: npt.NDArray) -> ma.MaskedArray:
    """Converts relative error to dB, i.e. 10 log10(1 + error)."""
    return ma.array(utils.lin2db(1 + ma.array(rel_error)))


def _add_extinction_comments(attributes: dict, od_source: OpticalDepthSource) -> dict:
    comment = attributes["extinction_liquid"].comment.format(
        der=od_source.assumed_der * 1e6,
        der_min=DER_VALID_RANGE[0] * 1e6,
        der_max=DER_VALID_RANGE[1] * 1e6,
    )
    attributes["extinction_liquid"] = attributes["extinction_liquid"]._replace(
        comment=comment
    )
    comment = attributes["extinction_liquid_error"].comment.format(
        der_error=ASSUMED_DER_REL_ERROR * 100
    )
    attributes["extinction_liquid_error"] = attributes[
        "extinction_liquid_error"
    ]._replace(comment=comment)
    return attributes


COMMENTS = {
    "extinction_liquid": (
        "This variable was calculated for the pixels where the categorization\n"
        "data has diagnosed liquid droplets and a reliable liquid water path was\n"
        "available from a coincident microwave radiometer. Where the liquid\n"
        "layer was not detected by the radar, or the retrieved droplet\n"
        "effective radius was outside the range {der_min:.0f}-{der_max:.0f} um\n"
        "(e.g. due to drizzle), an assumed effective radius of {der:.0f} um\n"
        "was used. Missing values indicate that liquid water path was\n"
        "unavailable, unreliable or zero, or that rain was present in the\n"
        "profile.\n"
        "Note that the liquid water path is distributed over the detected\n"
        "liquid layers only. If liquid is present in layers not classified as\n"
        "droplets (e.g. mixed-phase cloud above the lidar-detected liquid\n"
        "base), the extinction of the detected layers is overestimated. Such\n"
        "profiles are flagged in the optical_depth_retrieval_status variable."
    ),
    "extinction_ice": (
        "This variable was calculated for the pixels where the categorization\n"
        "data has diagnosed that the radar echo is due to ice. Missing values\n"
        "indicate that ice was detected only by the lidar, or that the radar\n"
        "reflectivity was affected by uncorrected liquid, rain or melting\n"
        "attenuation."
    ),
    "extinction_liquid_error": (
        "Random error in liquid extinction, one standard deviation, expressed\n"
        "as 10 log10(1 + relative error). It combines the liquid water content\n"
        "error (liquid water path error and cloud boundary uncertainty, see the\n"
        "lwc product) with the droplet effective radius error from the Frisch\n"
        "method, or an assumed {der_error:.0f} % where the effective radius\n"
        "was assumed."
    ),
    "extinction_ice_error": (
        "Random error in ice extinction, one standard deviation, expressed as\n"
        "10 log10(1 + relative error). The error of the ice water content\n"
        "retrieval (see the iwc product) is used, as the empirical\n"
        "reflectivity-temperature relation for extinction has a similar\n"
        "uncertainty. It includes the additional uncertainty of the liquid\n"
        "attenuation correction where applied."
    ),
    "optical_depth_error": (
        "Random error in cloud optical depth, one standard deviation, expressed\n"
        "as 10 log10(1 + relative error). The extinction errors are assumed\n"
        "fully correlated within a profile for each phase, as they are\n"
        "dominated by the liquid water path, effective radius and\n"
        "reflectivity-temperature relation uncertainties, and independent\n"
        "between liquid and ice. Systematic errors, such as liquid in layers\n"
        "not classified as droplets or ice detected only by the lidar, are\n"
        "not included; see the retrieval status."
    ),
    "optical_depth": (
        "Vertical integral of extinction over the profile. The value is zero\n"
        "in profiles where no cloud was detected. The retrieval is not\n"
        "performed, and the value is missing, if rain is present, if liquid\n"
        "cloud is present but liquid water path is unavailable or unreliable,\n"
        "or if ice is present but affected by uncorrected radar attenuation.\n"
        "Pixels where ice was detected only by the lidar do not contribute,\n"
        "so the value may be a lower bound; see the retrieval status."
    ),
}

DEFINITIONS = {
    "extinction_retrieval_status": utils.status_field_definition(
        {
            0: """No cloud detected.""",
            1: """Liquid: reliable retrieval with radar-based droplet
                  effective radius.""",
            2: """Liquid: droplet effective radius not retrieved from radar,
                  assumed value used.""",
            3: """Ice: reliable retrieval.""",
            4: """Ice: retrieval performed with radar corrected for liquid,
                  rain or melting attenuation.""",
            5: """Mixed phase: both liquid and ice extinction retrieved
                  in the same pixel.""",
            6: """Cloud detected but no retrieval: rain, missing or
                  non-positive liquid water path, ice detected only by lidar,
                  or uncorrected radar attenuation.""",
        }
    ),
    "optical_depth_retrieval_status": utils.status_field_definition(
        {
            0: """Clear sky: no cloud detected, optical depth is zero.""",
            1: """Reliable retrieval.""",
            2: """Assumed droplet effective radius used where the radar-based
                  retrieval was unavailable or outside the valid range.""",
            3: """Ice detected only by lidar in part of the profile: ice
                  optical depth is a lower bound.""",
            4: """Liquid water path exceeds the adiabatic liquid water path of
                  the detected liquid layers by more than a factor of two:
                  liquid is probably present in layers not classified as
                  droplets, and the vertical distribution of liquid
                  extinction is unreliable.""",
            5: """No retrieval: rain, missing or unreliable liquid water path,
                  or uncorrected radar attenuation in ice.""",
        }
    ),
}

OPTICAL_DEPTH_ATTRIBUTES = {
    "extinction_liquid": MetaData(
        long_name="Visible extinction coefficient of liquid cloud",
        units="m-1",
        ancillary_variables="extinction_liquid_error",
        comment=COMMENTS["extinction_liquid"],
        dimensions=("time", "height"),
    ),
    "extinction_liquid_error": MetaData(
        long_name="Random error in liquid extinction coefficient",
        units="dB",
        comment=COMMENTS["extinction_liquid_error"],
        dimensions=("time", "height"),
    ),
    "extinction_ice": MetaData(
        long_name="Visible extinction coefficient of ice cloud",
        units="m-1",
        ancillary_variables="extinction_ice_error",
        comment=COMMENTS["extinction_ice"],
        dimensions=("time", "height"),
    ),
    "extinction_ice_error": MetaData(
        long_name="Random error in ice extinction coefficient",
        units="dB",
        comment=COMMENTS["extinction_ice_error"],
        dimensions=("time", "height"),
    ),
    "optical_depth_error": MetaData(
        long_name="Random error in cloud optical depth",
        units="dB",
        comment=COMMENTS["optical_depth_error"],
        dimensions=("time",),
    ),
    "extinction_retrieval_status": MetaData(
        long_name="Extinction coefficient retrieval status",
        definition=DEFINITIONS["extinction_retrieval_status"],
        units="1",
        dimensions=("time", "height"),
    ),
    "optical_depth_liquid": MetaData(
        long_name="Liquid cloud optical depth",
        units="1",
        comment=COMMENTS["optical_depth"],
        dimensions=("time",),
    ),
    "optical_depth_ice": MetaData(
        long_name="Ice cloud optical depth",
        units="1",
        comment=COMMENTS["optical_depth"],
        dimensions=("time",),
    ),
    "optical_depth": MetaData(
        long_name="Cloud optical depth",
        standard_name="atmosphere_optical_thickness_due_to_cloud",
        units="1",
        ancillary_variables="optical_depth_error optical_depth_retrieval_status",
        comment=COMMENTS["optical_depth"],
        dimensions=("time",),
    ),
    "optical_depth_retrieval_status": MetaData(
        long_name="Cloud optical depth retrieval status",
        definition=DEFINITIONS["optical_depth_retrieval_status"],
        units="1",
        dimensions=("time",),
    ),
}
