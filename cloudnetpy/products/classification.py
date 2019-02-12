import numpy as np
import cloudnetpy.utils as utils
import cloudnetpy.output as output
from cloudnetpy.categorize import DataSource
from cloudnetpy.products.ncf import save_Cnet

class DataCollect(DataSource):
    def __init__(self, cat_file):
        super().__init__(cat_file)
        self.height = self._getvar('height')

def generate_class(cat_file):
    data = DataCollect(cat_file)
    vrs = data.variables

    target_classification = class_masks(vrs['category_bits'][:])
    status = class_status(vrs['quality_bits'][:])
    cloud_mask, base_height, top_height = cloud_layer_heights(target_classification, vrs['height'])

    class2cnet(data, {'classification_pixels':target_classification,
                      'classification_quality_pixels':status,
                      'cloud_mask':cloud_mask,
                      'cloud_bottom':base_height,
                      'cloud_top':top_height})


def class2cnet(data, vars_in):
    """ Defines Classification Cloudnet objects """
    classification_data = {}
    i = 0
    for key,value in vars_in.items():
        data.append_data(value, key)
        i += 1
    output.update_attributes(data.data)
    save_Cnet(data, 'test_class2.nc', 'Classification', 0.1)


def cloud_layer_heights(target_classification, height):
    cloud_mask = np.zeros_like(target_classification, dtype=int)

    for i in range(np.max(target_classification)):
        if i == 1 or i == 3 or i == 4 or i == 5:
            cloud_mask[target_classification == i] = 1

    base_height = top_height = np.full((len(cloud_mask),), np.nan)

    for ii in range(len(cloud_mask)):
        if cloud_mask[ii, :].any == 1:
            inds = np.argwhere(cloud_mask[ii, :] == 1)
            base_height[ii] = height[inds[0]]
            top_height[ii] = height[inds[-1]]

    return (cloud_mask, base_height, top_height)


def class_bits(cb, keys):
    bits = {}

    for i in range(len(keys)):
        bits[keys[i]] = utils.isbit(cb, i)

    return bits


def class_status(qb):
    keys = ['radar_bit', 'lidar_bit', 'radar_clutter_bit', 'lidar_molecular_bit', 'radar_attenuated_bit', 'radar_corrected_bit']
    q_bits = class_bits(qb, keys)

    quality_mask = np.copy(q_bits['lidar_bit'].astype(int))
    quality_mask[q_bits['radar_attenuated_bit'] & q_bits['radar_corrected_bit'] & q_bits['radar_bit']] = 2
    quality_mask[q_bits['radar_bit'] & q_bits['lidar_bit']] = 3
    quality_mask[q_bits['radar_attenuated_bit'] & q_bits['radar_corrected_bit']] = 4
    quality_mask[q_bits['radar_bit']] = 5
    quality_mask[q_bits['radar_corrected_bit']] = 6
    quality_mask[q_bits['radar_corrected_bit'] & q_bits['radar_bit']] = 7
    quality_mask[q_bits['radar_clutter_bit']] = 8
    quality_mask[q_bits['lidar_molecular_bit'] & q_bits['radar_bit']] = 9

    return quality_mask


def class_masks(cb):
    keys = ['droplet_bit', 'falling_bit', 'cold_bit', 'melting_bit', 'aerosol_bit', 'insect_bit']
    bits = class_bits(cb, keys)

    ind = np.where(bits['falling_bit'] & bits['cold_bit'])
    target_classification = bits['droplet_bit'].astype(int) + 2 * bits['falling_bit'].astype(int)
    target_classification[ind] = target_classification[ind] + 1
    target_classification[bits['melting_bit']] = 6
    target_classification[bits['melting_bit'] & bits['droplet_bit']] = 7
    target_classification[bits['aerosol_bit']] = 8
    target_classification[bits['insect_bit']] = 9
    target_classification[bits['aerosol_bit'] & bits['insect_bit']] = 10

    return target_classification


