"""Module for creating classification file."""
import numpy as np
import cloudnetpy.output as output
from cloudnetpy.categorize import DataSource
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import CategorizeBits


def generate_class(categorize_file, output_file):
    """Generates Cloudnet classification product.

    Generates categorized bins to 10 types of different targets in atmosphere
    as well as instrument status classification. Classifications are saved to
    NetCDF file with information of classification and measurements.

    Args:
        categorize_file (str): Categorize file name.

        output_file (str): Output file name.

    Examples:
        >>> from cloudnetpy.products.classification import generate_class
        >>> generate_class('categorize.nc', 'classification.nc')

    """
    data_handler = DataSource(categorize_file)
    categorize_bits = CategorizeBits(categorize_file)
    classification = get_target_classification(categorize_bits)
    data_handler.append_data(classification, 'target_classification')
    status = get_detection_status(categorize_bits)
    data_handler.append_data(status, 'detection_status')
    output.update_attributes(data_handler.data, CLASSIFICATION_ATTRIBUTES)
    output.save_product_file('classification', data_handler, output_file)


def get_target_classification(categorize_bits):
    """Classifies atmospheric targets by combining active bits."""
    bits = categorize_bits.category_bits
    classification = bits['droplet'] + 2*bits['falling']  # 0, 1, 2, 3
    classification[bits['falling'] & bits['cold']] += 2  # 4, 5
    classification[bits['melting']] = 6
    classification[bits['melting'] & bits['droplet']] = 7
    classification[bits['aerosol']] = 8
    classification[bits['insect']] = 9
    classification[bits['aerosol'] & bits['insect']] = 10
    return classification


def get_detection_status(categorize_bits):
    """Classifies detection status by combining active bits."""
    bits = categorize_bits.quality_bits
    status = np.copy(bits['lidar'].astype(int))
    status[bits['radar']] = 5
    status[bits['lidar'] & bits['radar']] = 3
    status[bits['corrected']] = 6
    status[bits['corrected'] & bits['radar']] = 7
    status[bits['attenuated'] & ~bits['corrected']] = 4
    status[bits['attenuated'] & ~bits['corrected'] & bits['radar']] = 2
    status[bits['clutter']] = 8
    status[bits['molecular'] & ~bits['radar']] = 9
    return status


COMMENTS = {
    'target_classification':
        ('This variable is a simplification of the bitfield "category_bits" in the\n'
         'target categorization and data quality dataset. It provides the 9 main\n'
         'atmospheric target classifications that can be distinguished by radar and\n'
         'lidar. The classes are defined in the definition attributes.'),

    'detection_status':
        ('This variable is a simplification of the bitfield "quality_bits" in the\n'
         'target categorization and data quality dataset. It reports on the\n'
         'reliability of the radar and lidar data used to perform the classification.\n'
         'The classes are defined in the definition attributes.'),
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
         'Value 1: Lidar echo only.\n'
         'Value 2: Radar echo but reflectivity may be unreliable as attenuation by\n'
         '         rain, melting ice or liquid cloud has not been corrected.\n'
         'Value 3: Good radar and lidar echos.\n'
         'Value 4: No radar echo but rain or liquid cloud beneath mean that\n'
         '         attenuation that would be experienced is unknown.\n'
         'Value 5: Good radar echo only.\n'
         'Value 6: No radar echo but known attenuation.\n'
         'Value 7: Radar echo corrected for liquid cloud attenuation\n'
         '         using microwave radiometer data.\n'
         'Value 8: Radar ground clutter.\n'
         'Value 9: Lidar clear-air molecular scattering.'),
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
