""" This module is used to generate
categorize (Level 1) product from pre-processed
radar, lidar and MWR files.
"""

# import sys
# import numpy as np
import numpy.ma as ma
import utils.ncf as ncf


def generate_categorize(input_files, output_file, aux):
    """ Generate Cloudnet Level 1 categorize file.

    Args:
        input_files (tuple): Tuple of strings containing full paths of
                             4 input files (radar, lidar, mwr, model).
        output_file (str): Full path of output file.
        aux (tuple): Tuple of strings including some metadata
                     of the site (site_name, institute).

    """
    rad, rad_vrs = ncf.load_nc(input_files[0])
    lid, lid_vrs = ncf.load_nc(input_files[1])
    mwr, mwr_vrs = ncf.load_nc(input_files[2])
    mod, mod_vrs = ncf.load_nc(input_files[3])

    try:
        freq, is35 = get_radar_freq(rad_vrs)
    except (ValueError, KeyError) as error:
        print(error)


def get_radar_freq(vrs):
    """ Return frequency of radar.

    Args:
        vrs: Pointer to radar variables.

    Returns:
        Tuple containing

        - **frequency** (*float*): Frequency or radar.
        - **is35** (*boolean*): True if 35 GHz radar. False if 94 GHz.

    Raises:
        KeyError: No frequency in the radar file.
        ValueError: Invalid frequency value.

    """
    possible_fields = ('radar_frequency', 'frequency')  # Several possible
    freq = [vrs[field][:] for field in vrs if field in possible_fields]

    if not freq:
        raise KeyError('Missing frequency in the radar file.')
    else:
        freq = freq[0]  # actual data of the masked data

    assert ma.count(freq) == 1, 'Multiple frequencies. Not a radar file??'

    if 30 < freq < 40:
        is35 = True
    elif 90 < freq < 100:
        is35 = False
    else:
        raise ValueError('Only 35 and 94 GHz radas supported.')

    return float(freq), is35
