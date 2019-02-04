import numpy as np
import cloudnetpy.utils as utils
from cloudnetpy.categorize import RawDataSource
from cloudnetpy.metadata import _COMMENTS, _DEFINITIONS
from cloudnetpy.products.ncf import CnetVar

class DataCollect(RawDataSource):
    def __init__(self, cat_file):
        super().__init__(cat_file)
        # Lisätään tänne tarpeellisia juttuja, jos tulee tarvetta

def generate_class(cat_file):
    # Kutsutaan luokkaa DataCollect
    # Pitää tarkastaa onko parempaa tapaa implementoida
    data = DataCollect(cat_file)
    cat = data.dataset
    vrs = data.variables

    target_classification = class_masks(vrs['category_bits'][:])
    status = class_status(vrs['quality_bits'][:])
    cloud_mask, base_height, top_height = cloud_layer_heights(target_classification, vrs['height'])

    # Muutetaan tämä siten, että talletetaan suoraan data
    obs = class2cnet({'target_classification':target_classification,
                      'detection_status':status,
                      'cloud_mask':cloud_mask,
                      'cloud_base_height':base_height,
                      'cloud_top_height':top_height})

    return (cat, obs)


def class2cnet(vars_in):
    """ Defines Classification Cloudnet objects """
    #TODO: Tee tämä jossain vaiheessa valmiiksi, pitää siis hakea haluttu data metadatasta

    log, lin = ('logarithmic', 'linear')
    obs = []
    s, lname = ('target_classification', 'Target classification')
    obs.append(CnetVar(s, vars_in[s], data_type='b', fill_value=None, long_name=lname, plot_range=(0,10),
                       comment='classification_pixels' ,
                       extra_attributes={'definition':'definition'}))

    s, lname = ('detection_status', 'Radar and lidar detection status')
    obs.append(CnetVar(s, vars_in[s], long_name=lname, plot_range=(0, 9), fill_value=None,
                       comment=('comment'),
                       extra_attributes={'definition':'definition'}))

    s, lname = ('cloud_base_height', 'Height of cloud base above ground')
    obs.append(CnetVar(s, vars_in[s], size='time', long_name=lname, units='m',
                       comment='cloud_bottom'))
    s, lname = ('cloud_top_height', 'Height of cloud top above ground')
    obs.append(CnetVar(s, vars_in[s], size='time', long_name=lname, units='m',
                       comment='cloud_top'))

    return obs


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


