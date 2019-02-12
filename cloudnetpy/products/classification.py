import numpy as np
import cloudnetpy.utils as utils
import cloudnetpy.output as output
from cloudnetpy.categorize import DataSource
from cloudnetpy.products.ncf import save_Cnet

"""
Nyt tämä toimii periaatteessa oikein. Pitää vielä hoitaa seuraavat asiat:
- Muuta data paremman nimiseksi
- Muuta bit_class funktio paremman nimiseksi

-Tiedoston talletuspaikka jossain vaiheessa kuntoon, samoin nimi
- ncf tiedostoa pitää hinkata
- Dokumentaatio docstring ja testit pitää vielä kirjoittaa
"""

class DataCollect(DataSource):
    def __init__(self, cat_file):
        super().__init__(cat_file)
        self.height = self._getvar('height')

def generate_class(cat_file):
    data = DataCollect(cat_file)
    class_masks(data)
    class_status(data)
    cloud_layer_heights(data)

    output.update_attributes(data.data)
    save_Cnet(data, 'test_class2.nc', 'Classification', 0.1)


def cloud_layer_heights(data):
    #TODO: voisi miettiä, josko siistisi tätä
    target_classification = data['target_classification']
    height = data.dataset['height']

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

    data.append_data(cloud_mask, 'cloud_mask')
    data.append_data(base_height, 'cloud_bottom')
    data.append_data(top_height, 'cloud_top')


def class_bits(cb, keys):
    bits = {}

    for i in range(len(keys)):
        bits[keys[i]] = utils.isbit(cb, i)

    return bits


def class_status(data):
    qb = data.dataset['quality_bits'][:]

    keys = ['radar_bit', 'lidar_bit', 'radar_clutter_bit', 'lidar_molecular_bit',
            'radar_attenuated_bit', 'radar_corrected_bit']
    q_bits = class_bits(qb, keys)

    quality_mask = np.copy(q_bits['lidar_bit'].astype(int))
    quality_mask[q_bits['radar_attenuated_bit'] & q_bits['radar_corrected_bit']
                 & q_bits['radar_bit']] = 2
    quality_mask[q_bits['radar_bit'] & q_bits['lidar_bit']] = 3
    quality_mask[q_bits['radar_attenuated_bit'] & q_bits['radar_corrected_bit']] = 4
    quality_mask[q_bits['radar_bit']] = 5
    quality_mask[q_bits['radar_corrected_bit']] = 6
    quality_mask[q_bits['radar_corrected_bit'] & q_bits['radar_bit']] = 7
    quality_mask[q_bits['radar_clutter_bit']] = 8
    quality_mask[q_bits['lidar_molecular_bit'] & q_bits['radar_bit']] = 9

    data.append_data(quality_mask, 'quality_mask')


def class_masks(data):
    cb = data.dataset['category_bits'][:]

    keys = ['droplet_bit', 'falling_bit', 'cold_bit', 'melting_bit',
            'aerosol_bit', 'insect_bit']
    bits = class_bits(cb, keys)

    ind = np.where(bits['falling_bit'] & bits['cold_bit'])
    target_classification = bits['droplet_bit'].astype(int) + \
                            2 * bits['falling_bit'].astype(int)
    target_classification[ind] = target_classification[ind] + 1
    target_classification[bits['melting_bit']] = 6
    target_classification[bits['melting_bit'] & bits['droplet_bit']] = 7
    target_classification[bits['aerosol_bit']] = 8
    target_classification[bits['insect_bit']] = 9
    target_classification[bits['aerosol_bit'] & bits['insect_bit']] = 10

    data.append_data(target_classification, 'target_classification')



