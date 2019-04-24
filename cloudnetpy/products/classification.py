"""Module for creating classification file."""
import numpy as np
import cloudnetpy.output as output
from cloudnetpy.categorize import DataSource
import cloudnetpy.products.product_tools as p_tools
from cloudnetpy.metadata import CLASSIFICATION_ATTRIBUTES
from cloudnetpy.products.product_tools import CategorizeBits


def generate_class(categorize_file, output_file):
    """High level API to generate Cloudnet classification product.

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
    _save_data_and_meta(data_handler, output_file)


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


def _save_data_and_meta(data_handler, output_file):
    """
    Saves wanted information to NetCDF file.
    """
    dims = {'time': len(data_handler.time),
            'height': len(data_handler.variables['height'])}
    rootgrp = output.init_file(output_file, dims, data_handler.data, zlib=True)
    vars_from_source = ('altitude', 'latitude', 'longitude', 'time', 'height')
    output.copy_variables(data_handler.dataset, rootgrp, vars_from_source)
    rootgrp.title = f"Classification file from {data_handler.dataset.location}"
    rootgrp.source = f"Categorize file: {p_tools.get_source(data_handler)}"
    output.copy_global(data_handler.dataset, rootgrp, ('location', 'day',
                                                       'month', 'year'))
    output.merge_history(rootgrp, 'classification', data_handler)
    rootgrp.close()
