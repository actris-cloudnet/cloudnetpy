"""Module for creating classification file."""
from typing import Union
import numpy as np
from cloudnetpy import output
from cloudnetpy.categorize import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import CategorizeBits


def generate_classification(categorize_file: str,
                            output_file: str,
                            keep_uuid: bool = False,
                            uuid: Union[str, None] = None) -> str:
    """Generates Cloudnet classification product.

    This function reads the initial classification masks from a
    categorize file and creates a more comprehensive classification
    for different atmospheric targets. The results are written in a
    netCDF file.

    Args:
        categorize_file (str): Categorize file name.
        output_file (str): Output file name.
        keep_uuid (bool, optional): If True, keeps the UUID of the old file,
            if that exists. Default is False when new UUID is generated.
        uuid (str, optional): Set specific UUID for the file.

    Returns:
        str: UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_classification
        >>> generate_classification('categorize.nc', 'classification.nc')

    """
    data_handler = DataSource(categorize_file)
    categorize_bits = CategorizeBits(categorize_file)
    classification = _get_target_classification(categorize_bits)
    data_handler.append_data(classification, 'target_classification')
    status = _get_detection_status(categorize_bits)
    data_handler.append_data(status, 'detection_status')
    date = data_handler.get_date()
    attributes = output.add_time_attribute(CLASSIFICATION_ATTRIBUTES, date)
    output.update_attributes(data_handler.data, attributes)
    uuid = output.save_product_file('classification', data_handler, output_file, keep_uuid, uuid)
    data_handler.close()
    return uuid


def _get_target_classification(categorize_bits):
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


def _get_detection_status(categorize_bits):
    bits = categorize_bits.quality_bits
    status = np.zeros(bits['radar'].shape, dtype=int)
    status[bits['radar'] & bits['lidar']] = 1
    status[bits['radar'] & ~bits['lidar'] & ~bits['attenuated']] = 2
    status[bits['radar'] & bits['corrected']] = 4
    status[bits['lidar'] & ~bits['radar']] = 3
    status[bits['radar'] & bits['attenuated'] & ~bits['corrected']] = 5
    status[bits['clutter']] = 6
    status[bits['molecular'] & ~bits['radar']] = 7
    return status


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
    )
}
