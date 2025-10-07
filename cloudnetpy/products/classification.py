"""Module for creating classification file."""

from enum import IntEnum
from os import PathLike
from typing import NamedTuple
from uuid import UUID

import numpy as np
import numpy.typing as npt
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.categorize import atmos_utils
from cloudnetpy.constants import M_S_TO_MM_H
from cloudnetpy.datasource import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import CategorizeBits, QualityBits


def generate_classification(
    categorize_file: str | PathLike,
    output_file: str | PathLike,
    uuid: str | UUID | None = None,
) -> UUID:
    """Generates Cloudnet classification product.

    This function reads the initial classification masks from a
    categorize file and creates a more comprehensive classification
    for different atmospheric targets. The results are written in a
    netCDF file.

    Args:
        categorize_file: Categorize file name.
        output_file: Output file name.
        uuid: Set specific UUID for the file.

    Returns:
        str: UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_classification
        >>> generate_classification('categorize.nc', 'classification.nc')

    """
    uuid = utils.get_uuid(uuid)
    categorize_bits = CategorizeBits(categorize_file)
    with DataSource(categorize_file) as source:
        classification = _get_target_classification(categorize_bits)
        source.append_data(classification, "target_classification")

        detection_status = _get_detection_status(categorize_bits)
        source.append_data(detection_status, "detection_status")

        signal_source_status = _get_signal_source_status(categorize_bits)
        source.append_data(signal_source_status, "signal_source_status")

        att_status = _get_radar_attenuation_status(source, categorize_bits)
        source.append_data(att_status, "radar_attenuation_status")

        height = source.getvar("height")
        bases, tops = _get_cloud_base_and_top_heights(classification, height)
        source.append_data(bases, "cloud_base_height_amsl")
        source.append_data(tops, "cloud_top_height_amsl")
        source.append_data(
            bases - source.altitude,
            "cloud_base_height_agl",
        )
        source.append_data(
            tops - source.altitude,
            "cloud_top_height_agl",
        )

        cloud_top_status = _get_cloud_top_height_status(source, tops, att_status)
        source.append_data(cloud_top_status, "cloud_top_height_status")

        date = source.get_date()
        attributes = output.add_time_attribute(CLASSIFICATION_ATTRIBUTES, date)
        output.update_attributes(source.data, attributes)
        file_type = "classification"
        if "liquid_prob" in source.dataset.variables:
            file_type += "-voodoo"
        output.save_product_file(
            file_type, source, output_file, uuid, copy_from_cat=("rain_detected",)
        )
        return uuid


class TopStatus(IntEnum):
    RELIABLE = 0
    MODERATE_ATT = 1
    UNCORR_ATT = 2
    SEVERE_ATT = 3
    ABOVE_RANGE = 4


class AttStatus(IntEnum):
    CLEAR = 0
    NEGLIGIBLE = 1
    SMALL = 2
    MODERATE = 3
    SEVERE = 4
    UNCORRECTED = 5
    UNDETECTED = 6


class SignalStatus(IntEnum):
    CLEAR = 0
    BOTH = 1
    RADAR_ONLY = 2
    LIDAR_ONLY = 3


class Target(IntEnum):
    CLEAR = 0
    DROPLET = 1
    DRIZZLE_OR_RAIN = 2
    DRIZZLE_OR_RAIN_AND_DROPLET = 3
    ICE = 4
    ICE_AND_SUPERCOOLED = 5
    MELTING = 6
    MELTING_AND_DROPLET = 7
    AEROSOL = 8
    INSECT = 9
    INSECT_AND_AEROSOL = 10


class DetectionStatus(IntEnum):
    CLEAR = 0
    LIDAR_ONLY = 1
    RADAR_UNCERTAIN_ATT = 2
    RADAR_AND_LIDAR = 3
    NO_RADAR_UNCERTAIN_ATT = 4
    RADAR_ONLY = 5
    NO_RADAR_KNOWN_ATT = 6
    RADAR_ATT_CORRECTED = 7
    CLUTTER = 8
    MOLECULAR_SCATT = 9


class AttenuationClass(NamedTuple):
    small: npt.NDArray
    moderate: npt.NDArray
    severe: npt.NDArray


def _get_cloud_top_height_status(
    product_container: DataSource, tops: npt.NDArray, att_status: npt.NDArray
) -> npt.NDArray:
    height = product_container.dataset.variables["height"][:]
    dist = np.abs(height[None, :] - tops[:, None])
    height_inds = dist.argmin(axis=1)
    att_at_top = att_status[np.arange(att_status.shape[0]), height_inds]
    status = np.zeros(att_at_top.size, dtype=int)
    status[att_at_top == AttStatus.MODERATE] = TopStatus.MODERATE_ATT
    status[att_at_top == AttStatus.SEVERE] = TopStatus.SEVERE_ATT
    status[att_at_top == AttStatus.UNCORRECTED] = TopStatus.UNCORR_ATT
    status[tops >= height[-1]] = TopStatus.ABOVE_RANGE
    return status


def _get_radar_attenuation_status(
    data_source: DataSource, categorize_bits: CategorizeBits
) -> npt.NDArray:
    bits = categorize_bits.quality_bits
    is_attenuated = _get_is_attenuated_mask(bits)
    is_corrected = _get_is_corrected_mask(bits)
    att = _get_attenuation_classes(data_source)
    severity = np.zeros_like(att.small, dtype=int)
    severity[bits.radar] = AttStatus.NEGLIGIBLE
    severity[att.small & bits.radar] = AttStatus.SMALL
    severity[att.moderate & bits.radar] = AttStatus.MODERATE
    severity[att.severe & bits.radar] = AttStatus.SEVERE
    severity[~is_corrected & is_attenuated & bits.radar] = AttStatus.UNCORRECTED
    is_severe = severity == AttStatus.SEVERE
    above_severe = utils.ffill(is_severe)
    severity[above_severe & ~is_severe] = AttStatus.UNDETECTED
    return severity


def _get_attenuation_classes(data_source: DataSource) -> AttenuationClass:
    def _read_atten(key: str) -> npt.NDArray:
        if key not in data_source.dataset.variables:
            return np.zeros(data_source.time.shape)
        data = data_source.getvar(key)
        if isinstance(data, ma.MaskedArray):
            return data.filled(0)
        return data

    liquid_atten = _read_atten("radar_liquid_atten")
    rain_atten = _read_atten("radar_rain_atten")
    melting_atten = _read_atten("radar_melting_atten")

    not_w_band = data_source.getvar("radar_frequency") < 90

    if "lwp" not in data_source.dataset.variables or not_w_band:
        lwp = np.zeros(data_source.time.shape)
    else:
        lwp_data = data_source.getvar("lwp")
        lwp = lwp_data.filled(0) if isinstance(lwp_data, ma.MaskedArray) else lwp_data

    if "rainfall_rate" not in data_source.dataset.variables or not_w_band:
        rain_rate = np.zeros(data_source.time.shape)
    else:
        rain_data = data_source.getvar("rainfall_rate") * M_S_TO_MM_H
        rain_rate = (
            rain_data.filled(0) if isinstance(rain_data, ma.MaskedArray) else rain_data
        )

    total_atten = liquid_atten + rain_atten + melting_atten

    threshold_moderate = 10  # dB
    threshold_severe = 15  # dB
    threshold_lwp = 1  # kg/m2
    threshold_rain = 3  # mm/h

    small = total_atten > 0
    moderate = total_atten >= threshold_moderate
    severe = (
        (total_atten > threshold_severe)
        | (lwp[:, np.newaxis] > threshold_lwp)
        | (rain_rate[:, np.newaxis] > threshold_rain)
    )

    return AttenuationClass(small=small, moderate=moderate, severe=severe)


def _get_target_classification(
    categorize_bits: CategorizeBits,
) -> ma.MaskedArray:
    bits = categorize_bits.category_bits
    clutter = categorize_bits.quality_bits.clutter
    classification = ma.zeros(bits.freezing.shape, dtype=int)
    classification[bits.droplet & ~bits.falling] = Target.DROPLET
    classification[~bits.droplet & bits.falling] = Target.DRIZZLE_OR_RAIN
    classification[bits.droplet & bits.falling] = Target.DRIZZLE_OR_RAIN_AND_DROPLET
    classification[~bits.droplet & bits.falling & bits.freezing] = Target.ICE
    classification[bits.droplet & bits.falling & bits.freezing] = (
        Target.ICE_AND_SUPERCOOLED
    )
    classification[bits.melting] = Target.MELTING
    classification[bits.melting & bits.droplet] = Target.MELTING_AND_DROPLET
    classification[bits.aerosol] = Target.AEROSOL
    classification[bits.insect & ~clutter] = Target.INSECT
    classification[bits.aerosol & bits.insect & ~clutter] = Target.INSECT_AND_AEROSOL
    classification[clutter & ~bits.aerosol] = Target.CLEAR
    return classification


def _get_detection_status(categorize_bits: CategorizeBits) -> npt.NDArray:
    bits = categorize_bits.quality_bits
    is_attenuated = _get_is_attenuated_mask(bits)
    is_corrected = _get_is_corrected_mask(bits)

    status = np.zeros(bits.radar.shape, dtype=int)
    status[bits.lidar & ~bits.radar] = DetectionStatus.LIDAR_ONLY
    status[bits.radar & bits.lidar] = DetectionStatus.RADAR_AND_LIDAR
    status[~bits.radar & is_attenuated & ~is_corrected] = (
        DetectionStatus.NO_RADAR_UNCERTAIN_ATT
    )
    status[bits.radar & ~bits.lidar & ~is_attenuated] = DetectionStatus.RADAR_ONLY
    status[~bits.radar & is_attenuated & is_corrected] = (
        DetectionStatus.NO_RADAR_KNOWN_ATT
    )
    status[bits.radar & is_corrected] = DetectionStatus.RADAR_ATT_CORRECTED
    status[bits.radar & is_attenuated & ~is_corrected] = (
        DetectionStatus.RADAR_UNCERTAIN_ATT
    )
    status[bits.clutter] = DetectionStatus.CLUTTER
    status[bits.molecular & ~bits.radar] = DetectionStatus.MOLECULAR_SCATT
    return status


def _get_is_corrected_mask(bits: QualityBits) -> npt.NDArray:
    is_attenuated = _get_is_attenuated_mask(bits)
    return (
        is_attenuated
        & (~bits.attenuated_liquid | bits.corrected_liquid)
        & (~bits.attenuated_rain | bits.corrected_rain)
        & (~bits.attenuated_melting | bits.corrected_melting)
    )


def _get_is_attenuated_mask(bits: QualityBits) -> npt.NDArray:
    return bits.attenuated_liquid | bits.attenuated_rain | bits.attenuated_melting


def _get_signal_source_status(categorize_bits: CategorizeBits) -> npt.NDArray:
    bits = categorize_bits.quality_bits
    status = np.zeros(bits.radar.shape, dtype=int)
    status[bits.radar & bits.lidar] = SignalStatus.BOTH
    status[bits.radar & ~bits.lidar] = SignalStatus.RADAR_ONLY
    status[bits.lidar & ~bits.radar] = SignalStatus.LIDAR_ONLY
    return status


def _get_cloud_base_and_top_heights(
    classification: npt.NDArray,
    height: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    cloud_mask = _find_cloud_mask(classification)
    if not cloud_mask.any():
        return ma.masked_all(cloud_mask.shape[0]), ma.masked_all(cloud_mask.shape[0])
    lowest_bases = atmos_utils.find_lowest_cloud_bases(cloud_mask, height)
    highest_tops = atmos_utils.find_highest_cloud_tops(cloud_mask, height)
    if not (highest_tops - lowest_bases >= 0).all():
        msg = "Cloud base higher than cloud top!"
        raise ValueError(msg)
    return lowest_bases, highest_tops


def _find_cloud_mask(classification: npt.NDArray) -> npt.NDArray:
    cloud_mask = np.zeros(classification.shape, dtype=int)
    for value in [
        Target.DROPLET,
        Target.DRIZZLE_OR_RAIN_AND_DROPLET,
        Target.ICE,
        Target.ICE_AND_SUPERCOOLED,
    ]:
        cloud_mask[classification == value] = 1
    return cloud_mask


COMMENTS = {
    "target_classification": (
        "\n"
        "This variable provides the main atmospheric target classifications\n"
        "that can be distinguished by radar and lidar."
    ),
    "detection_status": (
        "\n"
        "This variable reports on the reliability of the radar and lidar data\n"
        "used to perform the classification."
    ),
}


DEFINITIONS = {
    "target_classification": utils.status_field_definition(
        {
            Target.CLEAR: """Clear sky.""",
            Target.DROPLET: """Cloud liquid droplets only.""",
            Target.DRIZZLE_OR_RAIN: """Drizzle or rain.""",
            Target.DRIZZLE_OR_RAIN_AND_DROPLET: """Drizzle or rain
                coexisting with cloud liquid droplets.""",
            Target.ICE: """Ice particles.""",
            Target.ICE_AND_SUPERCOOLED: """Ice coexisting with
                supercooled liquid droplets.""",
            Target.MELTING: """Melting ice particles.""",
            Target.MELTING_AND_DROPLET: """Melting ice particles
                coexisting with cloud liquid droplets.""",
            Target.AEROSOL: """Aerosol particles, no cloud or precipitation.""",
            Target.INSECT: """Insects, no cloud or precipitation.""",
            Target.INSECT_AND_AEROSOL: """Aerosol coexisting
                with insects, no cloud or precipitation.""",
        }
    ),
    "detection_status": utils.status_field_definition(
        {
            DetectionStatus.CLEAR: """Clear sky.""",
            DetectionStatus.LIDAR_ONLY: """Lidar echo only.""",
            DetectionStatus.RADAR_UNCERTAIN_ATT: """
                Radar echo but reflectivity may be unreliable as attenuation
                by rain, melting ice or liquid cloud has not been
                corrected.""",
            DetectionStatus.RADAR_AND_LIDAR: """Good radar and lidar echos.""",
            DetectionStatus.NO_RADAR_UNCERTAIN_ATT: """
                No radar echo but rain or liquid cloud beneath mean that
                attenuation that would be experienced is unknown.""",
            DetectionStatus.RADAR_ONLY: """
                Good radar echo only.""",
            DetectionStatus.NO_RADAR_KNOWN_ATT: """
                No radar echo but known attenuation.""",
            DetectionStatus.RADAR_ATT_CORRECTED: """
                Radar echo corrected for liquid, rain or melting attenuation.""",
            DetectionStatus.CLUTTER: """
                Radar ground clutter.""",
            DetectionStatus.MOLECULAR_SCATT: """
                Lidar clear-air molecular scattering.""",
        }
    ),
    "cloud_top_height_status": utils.status_field_definition(
        {
            TopStatus.RELIABLE: """Reliable.""",
            TopStatus.MODERATE_ATT: """Uncertain due to moderate
                radar attenuation.""",
            TopStatus.UNCORR_ATT: """Uncertain due to incomplete
                radar attenuation correction.""",
            TopStatus.SEVERE_ATT: """Likely erroneous due to
                severe radar attenuation.""",
            TopStatus.ABOVE_RANGE: """Cloud top above radar
                measurement range.""",
        }
    ),
    "signal_source_status": utils.status_field_definition(
        {
            SignalStatus.CLEAR: """No signal from radar or lidar.""",
            SignalStatus.BOTH: """Signal from both radar and lidar.""",
            SignalStatus.RADAR_ONLY: """Signal from radar only.""",
            SignalStatus.LIDAR_ONLY: """Signal from lidar only.""",
        }
    ),
    "radar_attenuation_status": utils.status_field_definition(
        {
            AttStatus.CLEAR: """No radar signal.""",
            AttStatus.NEGLIGIBLE: """Radar signal,
                negligible attenuation (corrected).""",
            AttStatus.SMALL: """Radar signal,
                small attenuation (corrected).""",
            AttStatus.MODERATE: """Radar signal,
                moderate attenuation (corrected).""",
            AttStatus.SEVERE: """Radar signal,
                severe attenuation (corrected).""",
            AttStatus.UNCORRECTED: """Radar signal,
                attenuation present but not corrected.""",
            AttStatus.UNDETECTED: """No radar signal, cloud
                may be undetected due to severe attenuation beneath.""",
        }
    ),
}


CLASSIFICATION_ATTRIBUTES = {
    "target_classification": MetaData(
        long_name="Target classification",
        comment=COMMENTS["target_classification"],
        definition=DEFINITIONS["target_classification"],
        units="1",
        dimensions=("time", "height"),
    ),
    "detection_status": MetaData(
        long_name="Radar and lidar detection status",
        comment=COMMENTS["detection_status"],
        definition=DEFINITIONS["detection_status"],
        units="1",
        dimensions=("time", "height"),
    ),
    "signal_source_status": MetaData(
        long_name="Signal source status",
        units="1",
        dimensions=("time", "height"),
        definition=DEFINITIONS["signal_source_status"],
    ),
    "radar_attenuation_status": MetaData(
        long_name="Radar attenuation status",
        units="1",
        dimensions=("time", "height"),
        definition=DEFINITIONS["radar_attenuation_status"],
    ),
    "cloud_top_height_amsl": MetaData(
        long_name="Height of cloud top above mean sea level",
        units="m",
        dimensions=("time",),
        ancillary_variables="cloud_top_height_status",
    ),
    "cloud_base_height_amsl": MetaData(
        long_name="Height of cloud base above mean sea level",
        units="m",
        dimensions=("time",),
    ),
    "cloud_top_height_agl": MetaData(
        long_name="Height of cloud top above ground level",
        units="m",
        dimensions=("time",),
        ancillary_variables="cloud_top_height_status",
    ),
    "cloud_base_height_agl": MetaData(
        long_name="Height of cloud base above ground level",
        units="m",
        dimensions=("time",),
    ),
    "cloud_top_height_status": MetaData(
        long_name="Cloud top height quality status",
        units="1",
        dimensions=("time",),
        definition=DEFINITIONS["cloud_top_height_status"],
    ),
}
