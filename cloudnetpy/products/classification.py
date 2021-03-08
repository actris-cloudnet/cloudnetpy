"""Module for creating classification file."""
from typing import Tuple, Optional
import numpy as np
from cloudnetpy import output
from cloudnetpy.categorize import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import CategorizeBits
from cloudnetpy.categorize import atmos


def generate_classification(categorize_file: str,
                            output_file: str,
                            keep_uuid: bool = False,
                            uuid: Optional[str] = None) -> str:
    """Generates Cloudnet classification product.

    This function reads the initial classification masks from a
    categorize file and creates a more comprehensive classification
    for different atmospheric targets. The results are written in a
    netCDF file.

    Args:
        categorize_file: Categorize file name.
        output_file: Output file name.
        keep_uuid: If True, keeps the UUID of the old file, if that exists. Default is False when new UUID is generated.
        uuid: Set specific UUID for the file.

    Returns:
        str: UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_classification
        >>> generate_classification('categorize.nc', 'classification.nc')

    """
    product_container = DataSource(categorize_file)
    categorize_bits = CategorizeBits(categorize_file)
    classification = _get_target_classification(categorize_bits)
    product_container.append_data(classification, 'target_classification')
    status = _get_detection_status(categorize_bits)
    product_container.append_data(status, 'detection_status')
    bases, tops = _get_cloud_base_and_top_heights(classification, product_container)
    product_container.append_data(bases, 'cloud_base_height_amsl')
    product_container.append_data(tops, 'cloud_top_height_amsl')
    product_container.append_data(bases - product_container.altitude, 'cloud_base_height_agl')
    product_container.append_data(tops - product_container.altitude, 'cloud_top_height_agl')
    date = product_container.get_date()
    attributes = output.add_time_attribute(CLASSIFICATION_ATTRIBUTES, date)
    output.update_attributes(product_container.data, attributes)
    uuid = output.save_product_file('classification', product_container, output_file, keep_uuid, uuid)
    product_container.close()
    return uuid


def _get_target_classification(categorize_bits: CategorizeBits) -> np.ndarray:
    bits = categorize_bits.category_bits
    clutter = categorize_bits.quality_bits['clutter']
    classification = np.zeros(bits['cold'].shape, dtype=int)
    classification[bits['droplet'] & ~bits['falling']] = 1
    classification[~bits['droplet'] & bits['falling']] = 2
    classification[bits['droplet'] & bits['falling']] = 3
    classification[~bits['droplet'] & bits['falling'] & bits['cold']] = 4
    classification[bits['droplet'] & bits['falling'] & bits['cold']] = 5
    classification[bits['melting']] = 6
    classification[bits['melting'] & bits['droplet']] = 7
    classification[bits['aerosol']] = 8
    classification[bits['insect'] & ~clutter] = 9
    classification[bits['aerosol'] & bits['insect'] & ~clutter] = 10
    classification[clutter & ~bits['aerosol']] = 0
    return classification


def _get_detection_status(categorize_bits: CategorizeBits) -> np.ndarray:
    bits = categorize_bits.quality_bits
    status = np.zeros(bits['radar'].shape, dtype=int)
    status[bits['radar'] & bits['lidar']] = 1
    status[bits['radar'] & ~bits['lidar'] & ~bits['attenuated']] = 2
    status[bits['radar'] & bits['corrected']] = 3
    status[bits['lidar'] & ~bits['radar']] = 4
    status[bits['radar'] & bits['attenuated'] & ~bits['corrected']] = 5
    status[bits['clutter']] = 6
    status[bits['molecular'] & ~bits['radar']] = 7
    return status


def _get_cloud_base_and_top_heights(classification: np.ndarray,
                                    product_container: DataSource) -> Tuple[np.ndarray, np.ndarray]:
    height = product_container.getvar('height')
    cloud_mask = _find_cloud_mask(classification)
    lowest_bases = atmos.find_lowest_cloud_bases(cloud_mask, height)
    highest_tops = atmos.find_highest_cloud_tops(cloud_mask, height)
    assert (highest_tops - lowest_bases >= 0).all()
    return lowest_bases, highest_tops


def _find_cloud_mask(classification: np.ndarray) -> np.ndarray:
    cloud_mask = np.zeros(classification.shape, dtype=int)
    for value in [1, 3, 4, 5]:
        cloud_mask[classification == value] = 1
    return cloud_mask


COMMENTS = {
    'target_classification':
        ('This variable provides the main atmospheric target classifications\n'
         'that can be distinguished by radar and lidar.'),

    'detection_status':
        ('This variable reports on the reliability of the radar and lidar data\n'
         'used to perform the classification.')
}

DEFINITIONS = {
    'target_classification':
        ('\n'
         'Value 0: Clear sky.\n'
         'Value 1: Cloud liquid droplets only.\n'
         'Value 2: Drizzle or rain.\n'
         'Value 3: Drizzle or rain coexisting with cloud liquid droplets.\n'
         'Value 4: Ice particles.\n'
         'Value 5: Ice coexisting with supercooled liquid droplets.\n'
         'Value 6: Melting ice particles.\n'
         'Value 7: Melting ice particles coexisting with cloud liquid droplets.\n'
         'Value 8: Aerosol particles, no cloud or precipitation.\n'
         'Value 9: Insects, no cloud or precipitation.\n'
         'Value 10: Aerosol coexisting with insects, no cloud or precipitation.'),

    'detection_status':
        ('\n'
         'Value 0: Clear sky.\n'
         'Value 1: Good radar and lidar echos.\n'
         'Value 2: Good radar echo only.\n'
         'Value 3: Radar echo, corrected for liquid attenuation.\n'
         'Value 4: Lidar echo only.\n'
         'Value 5: Radar echo, uncorrected for liquid attenuation.\n'         
         'Value 6: Radar ground clutter.\n'
         'Value 7: Lidar clear-air molecular scattering.'),
}

CLASSIFICATION_ATTRIBUTES = {
    'target_classification': MetaData(
        long_name='Target classification',
        comment=COMMENTS['target_classification'],
        definition=DEFINITIONS['target_classification']
    ),
    'detection_status': MetaData(
        long_name='Radar and lidar detection status',
        comment=COMMENTS['detection_status'],
        definition=DEFINITIONS['detection_status']
    ),
    'cloud_top_height_amsl': MetaData(
        long_name='Height of cloud top above mean sea level',
        units='m',
    ),
    'cloud_base_height_amsl': MetaData(
        long_name='Height of cloud base above mean sea level',
        units='m',
    ),
    'cloud_top_height_agl': MetaData(
        long_name='Height of cloud top above ground level',
        units='m',
    ),
    'cloud_base_height_agl': MetaData(
        long_name='Height of cloud base above ground level',
        units='m',
    ),
}
