import sys
sys.path.insert(0, '/home/korpinen/Documents/ACTRIS/cloudnetpy/cloudnetpy')

import netCDF4
import numpy as np
from ncf import CnetVar
import utils

def generate_class(cat_file):
    cat = netCDF4.Dataset(cat_file)
    vrs = cat.variables

    target_classification = class_masks(vrs['category_bits'][:])
    status = class_status(vrs['quality_bits'][:])
    cloud_mask, base_height, top_height = cloud_layer_heights(target_classification, vrs['height'])

    obs = class2cnet({'target_classification':target_classification,
                      'detection_status':status,
                      'cloud_base_height':base_height,
                      'cloud_top_height':top_height})

    return (cat, obs)


def class2cnet(vars_in):
    """ Defines Classification Cloudnet objects """
    log, lin = ('logarithmic', 'linear')
    obs = []
    s, lname = ('target_classification', 'Target classification')
    obs.append(CnetVar(s, vars_in[s], data_type='b', fill_value=None, long_name=lname, plot_range=(0,10),
                       comment=('This variable is a simplification of the bitfield "category_bits" in the target categorization and data quality dataset.\n',
                     'It provides the 9 main atmospheric target classifications that can be distinguished by radar and lidar.\n',
                     'The classes are defined in the definition and long_definition attributes.'),
                       extra_attributes={'definition':('0: Clear sky',
                                                       '1: Cloud droplets only',
                                                       '2: Drizzle or rain',
                                                       '3: Drizzle/rain & cloud droplets',
                                                       '4: Ice',
                                                       '5: Ice & supercooled droplets',
                                                       '6: Melting ice',
                                                       '7: Melting ice & cloud droplets',
                                                       '8: Aerosol',
                                                       '9: Insects',
                                                       '10: Aerosol & insects'),
                                         'long_definition':('0: Clear sky',
                                                            '1: Cloud liquid droplets only',
                                                            '2: Drizzle or rain',
                                                            '3: Drizzle or rain coexisting with cloud liquid droplets',
                                                            '4: Ice particles',
                                                            '5: Ice coexisting with supercooled liquid droplets',
                                                            '6: Melting ice particles',
                                                            '7: Melting ice particles coexisting with cloud liquid droplets',
                                                            '8: Aerosol particles, no cloud or precipitation',
                                                            '9: Insects, no cloud or precipitation',
                                                            '10: Aerosol coexisting with insects, no cloud or precipitation'),
                                         'legend_key_red':(1, 0.4, 1, 0, 1, 0, 1, 0, 0.8, 0.6, 0.4),
                                         'legend_key_green':(1, 0.8, 0, 0, 0.9, 0.8, 0.6, 0.6, 0.8, 0.6, 0.4),
                                         'legend_key_blue':(1, 1, 0, 1, 0, 0, 0, 0.6, 0.8, 0.6, 0.4)}))

    s, lname = ('detection_status', 'Radar and lidar detection status')
    obs.append(CnetVar(s, vars_in[s], long_name=lname, plot_range=(0, 9), fill_value=None,
                       comment=('This variable is a simplification of the bitfield "quality_bits" in the target categorization and data quality dataset.',
                                'It reports on the reliability of the radar and lidar data used to perform the classification.',
                                'The classes are defined in the definition and long_definition attributes.'),
                       extra_attributes={'definition':('0: Clear sky',
                                                       '1: Lidar echo only',
                                                       '2: Radar echo but uncorrected atten.',
                                                       '3: Good radar & lidar echos',
                                                       '4: No radar but unknown attenuation',
                                                       '5: Good radar echo only',
                                                       '6: No radar but known attenuation',
                                                       '7: Radar corrected for liquid atten.',
                                                       '8: Radar ground clutter',
                                                       '9: Lidar molecular scattering'),
                                         'long_definition':('0: Clear sky',
                                                            '1: Lidar echo only',
                                                            '2: Radar echo but reflectivity may be unreliable as attenuation by rain, melting ice or liquid cloud has not been corrected',
                                                            '3: Good radar and lidar echos',
                                                            '4: No radar echo but rain or liquid cloud beneath mean that attenuation that would be experienced is unknown',
                                                            '5: Good radar echo only',
                                                            '6: No radar echo but known attenuation',
                                                            '7: Radar echo corrected for liquid cloud attenuation using microwave radiometer data',
                                                            '8: Radar ground clutter',
                                                            '9: Lidar clear-air molecular scattering'),
                                         'legend_key_red':(1, 1, 0.4, 0, 0.6, 0.4, 0.8, 0, 1, 1),
                                         'legend_key_green':(1, 0.9, 0.4, 0.8, 0.6, 0.8, 0.8, 0, 0, 0.6),
                                         'legend_key_blue':(1, 0, 0.4, 0, 0.6, 1, 0.8, 1, 0, 0)}))

    s, lname = ('cloud_base_height', 'Height of cloud base above ground')
    obs.append(CnetVar(s, vars_in[s], size='time', long_name=lname, units='m',
                       comment='This variable was calculated from the instance of cloud in the cloud mask variable and provides cloud base height for a maximum of 1 cloud layers'))
    s, lname = ('cloud_top_height', 'Height of cloud top above ground')
    obs.append(CnetVar(s, vars_in[s], size='time', long_name=lname, units='m',
                       comment='This variable was calculated from the instance of cloud in the cloud mask variable and provides cloud base top for a maximum of 1 cloud layers'))

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


