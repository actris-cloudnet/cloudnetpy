"""Module for creating classification file."""
import numpy as np
import cloudnetpy.output as output
from cloudnetpy.categorize import DataSource
import cloudnetpy.products.product_tools as p_tools
from cloudnetpy.metadata import CLASSIFICATION_ATTRIBUTES


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
    _append_target_classification(data_handler)
    _append_detection_status(data_handler)
    output.update_attributes(data_handler.data, CLASSIFICATION_ATTRIBUTES)
    _save_data_and_meta(data_handler, output_file)


def _append_detection_status(data_handler):
    """
    Makes classifications of instruments status by combining active bins
    """
    bits = p_tools.read_quality_bits(data_handler)

    quality = np.copy(bits['lidar'].astype(int))
    quality[bits['radar']] = 5
    quality[bits['lidar'] & bits['radar']] = 3
    quality[bits['corrected']] = 6
    quality[bits['corrected'] & bits['radar']] = 7
    quality[bits['attenuated'] & ~bits['corrected']] = 4
    quality[bits['attenuated'] & ~bits['corrected'] & bits['radar']] = 2
    quality[bits['clutter']] = 8
    quality[bits['molecular'] & ~bits['radar']] = 9

    data_handler.append_data(quality, 'detection_status')


def _append_target_classification(data_handler):
    """
    Makes classifications for the atmospheric targets by combining active bins
    """
    bits = p_tools.read_category_bits(data_handler)

    classification = bits['droplet'] + 2*bits['falling']  # 0, 1, 2, 3
    classification[bits['falling'] & bits['cold']] += 2  # 4, 5
    classification[bits['melting']] = 6
    classification[bits['melting'] & bits['droplet']] = 7
    classification[bits['aerosol']] = 8
    classification[bits['insect']] = 9
    classification[bits['aerosol'] & bits['insect']] = 10

    data_handler.append_data(classification, 'target_classification')


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


