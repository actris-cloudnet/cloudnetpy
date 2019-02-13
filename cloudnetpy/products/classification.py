import numpy as np
import cloudnetpy.utils as utils
import cloudnetpy.output as output
from cloudnetpy.categorize import DataSource

class DataCollect(DataSource):
    def __init__(self, cat_file):
        """ Class to generate and storage data and information

        Args:
            cat_file: NetCDf file of categorized bins and information of
                measurements and instruments
        """
        super().__init__(cat_file)
        self.height = self._getvar('height')

def generate_class(cat_file):
    """Makes classification for different types of targets at atmosphere.

    Generates categorized bins to 10 types of different targets in atmosphere
    as well as instrument status classification. Classifications are saved to
    NetCDF file with information of classification and measurements.

    Args:
        cat_file: NetCDf file of categorized bins and information of
                measurements and instruments
    """
    data_handler = DataCollect(cat_file)
    class_masks(data_handler)
    class_status(data_handler)
    #cloud_layer_heights(data_handler)
    output.update_attributes(data_handler.data)
    save_Cnet(data_handler, 'test_class2.nc', 'Classification', 0.1)


def cloud_layer_heights(data_handler):
    """
    Define cloudmask from categorized bins and finds hieghts of all clouds bases
    and tops
    """
    # Not in use
    target_classification = data_handler.data['target_classification'].data
    height = data_handler.dataset.variables['height']
    cloud_mask = np.zeros_like(target_classification, dtype=int)

    for i in range(np.max(target_classification)):
        if i == 1 or i == 3 or i == 4 or i == 5:
            cloud_mask[target_classification == i] = 1

    base_heights = np.full(target_classification.shape, np.nan)
    top_heights = np.full(target_classification.shape, np.nan)

    for i in range(len(cloud_mask[0])):
        base_height, top_height = utils.bases_and_tops(cloud_mask[i])
        base_heights[i][base_height] = height[base_height]
        top_heights[i][top_height] = height[top_height]

    data_handler.append_data(cloud_mask, 'cloud_mask')
    data_handler.append_data(base_heights, 'cloud_bottom')
    data_handler.append_data(top_heights, 'cloud_top')


def check_active_bits(cb, keys):
    """
    Check is observed bin active or not, returns boolean array of
    active and unactive bin index
    """
    bits = {}
    for i, key in enumerate(keys):
        bits[key] = utils.isbit(cb, i)
    return bits


def class_status(data_handler):
    """
    Makes classifications of instruments status by combining active bins
    """
    qb = data_handler.dataset['quality_bits'][:]

    keys = ['radar_bit', 'lidar_bit', 'radar_clutter_bit', 'lidar_molecular_bit',
            'radar_attenuated_bit', 'radar_corrected_bit']
    q_bits = check_active_bits(qb, keys)

    quality_mask = np.copy(q_bits['lidar_bit'])
    quality_mask[q_bits['radar_attenuated_bit'] & q_bits['radar_corrected_bit']
                 & q_bits['radar_bit']] = 2
    quality_mask[q_bits['radar_bit'] & q_bits['lidar_bit']] = 3
    quality_mask[q_bits['radar_attenuated_bit'] & q_bits['radar_corrected_bit']] = 4
    quality_mask[q_bits['radar_bit']] = 5
    quality_mask[q_bits['radar_corrected_bit']] = 6
    quality_mask[q_bits['radar_corrected_bit'] & q_bits['radar_bit']] = 7
    quality_mask[q_bits['radar_clutter_bit']] = 8
    quality_mask[q_bits['lidar_molecular_bit'] & q_bits['radar_bit']] = 9

    data_handler.append_data(quality_mask, 'detection_status')


def class_masks(data_handler):
    """
    Makes classifications for the atmospheric targets by combining active bins
    """
    cb = data_handler.dataset['category_bits'][:]

    keys = ['droplet_bit', 'falling_bit', 'cold_bit', 'melting_bit',
            'aerosol_bit', 'insect_bit']
    bits = check_active_bits(cb, keys)

    ind = np.where(bits['falling_bit'] & bits['cold_bit'])
    target_classification = bits['droplet_bit'] + 2 * bits['falling_bit']
    target_classification[ind] = target_classification[ind] + 1
    target_classification[bits['melting_bit']] = 6
    target_classification[bits['melting_bit'] & bits['droplet_bit']] = 7
    target_classification[bits['aerosol_bit']] = 8
    target_classification[bits['insect_bit']] = 9
    target_classification[bits['aerosol_bit'] & bits['insect_bit']] = 10

    data_handler.append_data(target_classification, 'target_classification')


def save_Cnet(data_handler, output_file, varname, version):
    """
    Saves wanted information to NetCDF file.
    """
    dims = {'time': len(data_handler.time),
            'height': len(data_handler.height)}
    rootgrp = output.init_file(output_file, dims, data_handler.data, zlib=True)
    output.copy_variables(data_handler.dataset, rootgrp, ('altitude', 'latitude', 'longitude', 'time', 'height'))
    rootgrp.title = f"Classification file from {data_handler.dataset.location}"
    #rootgrp.institution = 'Data processed at the Finnish Meteorological Institute.'
    #rootgrp.software_version = version
    #rootgrp.git_version = git_version()
    #rootgrp.file_uuid = str(uuid.uuid4().hex)
    output.copy_global(data_handler.dataset, rootgrp, ('location', 'day', 'month', 'year', 'source', 'history'))
    rootgrp.close()

