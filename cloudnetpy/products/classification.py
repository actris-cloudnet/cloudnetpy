import numpy as np
import cloudnetpy.utils as utils
from cloudnetpy.categorize import RawDataSource
from cloudnetpy.metadata import _COMMENTS, _DEFINITIONS
from cloudnetpy.products.ncf import CnetVar, save_Cnet

class DataCollect(RawDataSource):
    def __init__(self, cat_file):
        super().__init__(cat_file)
        self.height = self._getvar('height')
        # Lisätään tänne tarpeellisia juttuja, jos tulee tarvetta

def generate_class(cat_file):
    # Kutsutaan luokkaa DataCollect
    # Pitää tarkastaa onko parempaa tapaa implementoida
    data = DataCollect(cat_file)
    vrs = data.variables

    target_classification = class_masks(vrs['category_bits'][:])
    status = class_status(vrs['quality_bits'][:])
    cloud_mask, base_height, top_height = cloud_layer_heights(target_classification, vrs['height'])

    # Muutetaan tämä siten, että talletetaan suoraan data
    classification_data = class2cnet({'target_classification':target_classification,
                                      'detection_status':status,
                                      'cloud_mask':cloud_mask,
                                      'cloud_base_height':base_height,
                                      'cloud_top_height':top_height})

    # Lisätään tähän vielä datan talletus
    save_Cnet(data, classification_data, 'test_class.nc', 'Classification', 0.1)


def class2cnet(vars_in):
    """ Defines Classification Cloudnet objects """
    lname = ['Target classification', 'Radar and lidar detection status', 'Total area of clouds',
            'Height of cloud base above ground', 'Height of cloud top above ground']
    comments = ['classification_pixels', 'classification_quality_pixels',
                'cloud_mask', 'cloud_bottom', 'cloud_top']
    definitions = ['classification_pixels', 'classification_quality_pixels',
                   'cloud_mask', 'cloud_bottom', 'cloud_top']
    classification_data = {}
    i = 0
    for key,value in vars_in.items():
        if 'cloud' in key:
            unit = 'm'
            size = '1d'
            fill_f = True
            definition = 'None'
        else:
            unit = None
            size = '2d'
            fill_f = None
            definition = _DEFINITIONS[definitions[i]]
        if key == 'cloud_mask':
            size = '2d'

        classification_data[key] = CnetVar(key, value, size, fill_value=fill_f, long_name=lname[i], units=unit,
                            comment=_COMMENTS[comments[i]], extra_attributes={'definition': definition})
        i += 1

    return classification_data


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


