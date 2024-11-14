"""Module for creating classification file."""

import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.categorize import atmos_utils
from cloudnetpy.datasource import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import CategorizeBits


def generate_classification(
    categorize_file: str,
    output_file: str,
    uuid: str | None = None,
) -> str:
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
    with DataSource(categorize_file) as product_container:
        categorize_bits = CategorizeBits(categorize_file)
        classification = _get_target_classification(categorize_bits)
        product_container.append_data(classification, "target_classification")
        status = _get_detection_status(categorize_bits)
        product_container.append_data(status, "detection_status")
        bases, tops = _get_cloud_base_and_top_heights(classification, product_container)
        product_container.append_data(bases, "cloud_base_height_amsl")
        product_container.append_data(tops, "cloud_top_height_amsl")
        product_container.append_data(
            bases - product_container.altitude,
            "cloud_base_height_agl",
        )
        product_container.append_data(
            tops - product_container.altitude,
            "cloud_top_height_agl",
        )
        date = product_container.get_date()
        attributes = output.add_time_attribute(CLASSIFICATION_ATTRIBUTES, date)
        output.update_attributes(product_container.data, attributes)
        file_type = "classification"
        if "liquid_prob" in product_container.dataset.variables:
            file_type += "-voodoo"
        return output.save_product_file(
            file_type,
            product_container,
            output_file,
            uuid,
        )


def _get_target_classification(
    categorize_bits: CategorizeBits,
) -> ma.MaskedArray:
    bits = categorize_bits.category_bits
    clutter = categorize_bits.quality_bits.clutter
    classification = ma.zeros(bits.freezing.shape, dtype=int)
    classification[bits.droplet & ~bits.falling] = 1  # Cloud droplets
    classification[~bits.droplet & bits.falling] = 2  # Drizzle or rain
    classification[bits.droplet & bits.falling] = 3  # Drizzle or rain and droplets
    classification[~bits.droplet & bits.falling & bits.freezing] = 4  # ice
    classification[bits.droplet & bits.falling & bits.freezing] = 5  # ice + supercooled
    classification[bits.melting] = 6  # melting layer
    classification[bits.melting & bits.droplet] = 7  # melting + droplets
    classification[bits.aerosol] = 8  # aerosols
    classification[bits.insect & ~clutter] = 9  # insects
    classification[bits.aerosol & bits.insect & ~clutter] = 10  # insects + aerosols
    classification[clutter & ~bits.aerosol] = 0
    return classification


def _get_detection_status(categorize_bits: CategorizeBits) -> np.ndarray:
    bits = categorize_bits.quality_bits

    is_attenuated = (
        bits.attenuated_liquid | bits.attenuated_rain | bits.attenuated_melting
    )
    is_corrected = (
        is_attenuated
        & (~bits.attenuated_liquid | bits.corrected_liquid)
        & (~bits.attenuated_rain | bits.corrected_rain)
        & (~bits.attenuated_melting | bits.corrected_melting)
    )

    status = np.zeros(bits.radar.shape, dtype=int)
    status[bits.lidar & ~bits.radar] = 1
    status[bits.radar & bits.lidar] = 3
    status[~bits.radar & is_attenuated & ~is_corrected] = 4
    status[bits.radar & ~bits.lidar & ~is_attenuated] = 5
    status[~bits.radar & is_attenuated & is_corrected] = 6
    status[bits.radar & is_corrected] = 7
    status[bits.radar & is_attenuated & ~is_corrected] = 2
    status[bits.clutter] = 8
    status[bits.molecular & ~bits.radar] = 9
    return status


def _get_cloud_base_and_top_heights(
    classification: np.ndarray,
    product_container: DataSource,
) -> tuple[np.ndarray, np.ndarray]:
    height = product_container.getvar("height")
    cloud_mask = _find_cloud_mask(classification)
    if not cloud_mask.any():
        return ma.masked_all(cloud_mask.shape[0]), ma.masked_all(cloud_mask.shape[0])
    lowest_bases = atmos_utils.find_lowest_cloud_bases(cloud_mask, height)
    highest_tops = atmos_utils.find_highest_cloud_tops(cloud_mask, height)
    if not (highest_tops - lowest_bases >= 0).all():
        msg = "Cloud base higher than cloud top!"
        raise ValueError(msg)
    return lowest_bases, highest_tops


def _find_cloud_mask(classification: np.ndarray) -> np.ndarray:
    cloud_mask = np.zeros(classification.shape, dtype=int)
    for value in [1, 3, 4, 5]:
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
            0: "Clear sky.",
            1: "Cloud liquid droplets only.",
            2: "Drizzle or rain.",
            3: "Drizzle or rain coexisting with cloud liquid droplets.",
            4: "Ice particles.",
            5: "Ice coexisting with supercooled liquid droplets.",
            6: "Melting ice particles.",
            7: "Melting ice particles coexisting with cloud liquid droplets.",
            8: "Aerosol particles, no cloud or precipitation.",
            9: "Insects, no cloud or precipitation.",
            10: "Aerosol coexisting with insects, no cloud or precipitation.",
        }
    ),
    "detection_status": utils.status_field_definition(
        {
            0: """Clear sky.""",
            1: """Lidar echo only.""",
            2: """Radar echo but reflectivity may be unreliable as attenuation
                  by rain, melting ice or liquid cloud has not been
                  corrected.""",
            3: """Good radar and lidar echos.""",
            4: """No radar echo but rain or liquid cloud beneath mean that
                  attenuation that would be experienced is unknown.""",
            5: """Good radar echo only.""",
            6: """No radar echo but known attenuation.""",
            7: """Radar echo corrected for liquid, rain or melting
                  attenuation.""",
            8: """Radar ground clutter.""",
            9: """Lidar clear-air molecular scattering.""",
        }
    ),
}

CLASSIFICATION_ATTRIBUTES = {
    "target_classification": MetaData(
        long_name="Target classification",
        comment=COMMENTS["target_classification"],
        definition=DEFINITIONS["target_classification"],
        units="1",
    ),
    "detection_status": MetaData(
        long_name="Radar and lidar detection status",
        comment=COMMENTS["detection_status"],
        definition=DEFINITIONS["detection_status"],
        units="1",
    ),
    "cloud_top_height_amsl": MetaData(
        long_name="Height of cloud top above mean sea level",
        units="m",
    ),
    "cloud_base_height_amsl": MetaData(
        long_name="Height of cloud base above mean sea level",
        units="m",
    ),
    "cloud_top_height_agl": MetaData(
        long_name="Height of cloud top above ground level",
        units="m",
    ),
    "cloud_base_height_agl": MetaData(
        long_name="Height of cloud base above ground level",
        units="m",
    ),
}
