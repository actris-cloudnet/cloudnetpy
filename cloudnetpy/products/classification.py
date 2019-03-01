"""Module for creating classification file."""
import numpy as np
import cloudnetpy.output as output
from cloudnetpy.categorize import DataSource
import cloudnetpy.products.product_tools as p_tools


def generate_class(cat_file, output_file):
    """Makes classification for different types of targets at atmosphere.

    Generates categorized bins to 10 types of different targets in atmosphere
    as well as instrument status classification. Classifications are saved to
    NetCDF file with information of classification and measurements.

    Args:
        cat_file: NetCDF file of categorized bins and information of
                measurements and instruments.

        output_file(str): Output file name.

    Examples:
        >>> from cloudnetpy.products.classification import generate_class
        >>> generate_class('categorize.nc', 'classification.nc')

    """
    data_handler = DataSource(cat_file)
    _append_target_classification(data_handler)
    _append_detection_status(data_handler)
    output.update_attributes(data_handler.data)
    _save_data_and_meta(data_handler, output_file)


def _append_detection_status(data_handler):
    """
    Makes classifications of instruments status by combining active bins
    """
    quality_bits = data_handler.getvar('quality_bits')

    keys = p_tools.get_status_keys()
    bits = p_tools.check_active_bits(quality_bits, keys)

    quality = np.copy(bits['lidar'])
    quality[bits['attenuated'] & bits['corrected'] & bits['radar']] = 2
    quality[bits['radar'] & bits['lidar']] = 3
    quality[bits['attenuated'] & bits['corrected']] = 4
    quality[bits['radar']] = 5
    quality[bits['corrected']] = 6
    quality[bits['corrected'] & bits['radar']] = 7
    quality[bits['clutter']] = 8
    quality[bits['molecular'] & bits['radar']] = 9

    data_handler.append_data(quality, 'detection_status')


def _append_target_classification(data_handler):
    """
    Makes classifications for the atmospheric targets by combining active bins
    """
    category_bits = data_handler.getvar('category_bits')

    keys = p_tools.get_categorize_keys()
    bits = p_tools.check_active_bits(category_bits, keys)

    classification = bits['droplet'] + 2*bits['falling']

    falling_cold = np.where(bits['falling'] & bits['cold'])
    classification[falling_cold] += 2

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
    rootgrp.source = f"Categorize file: {_get_source(data_handler)}"
    output.copy_global(data_handler.dataset, rootgrp, ('location', 'day',
                                                       'month', 'year'))
    output.merge_history(rootgrp, 'classification', data_handler)
    rootgrp.close()


def _get_source(data_handler):
    """Returns uuid (or filename if uuid not found) of the source file."""
    return getattr(data_handler.dataset, 'file_uuid', data_handler.filename)

