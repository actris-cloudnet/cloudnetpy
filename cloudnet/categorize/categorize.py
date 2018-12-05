""" This module is used to generate
categorize (Level 1) product from pre-processed
radar, lidar and MWR files.
"""

# import sys
import numpy as np
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

    TIME_RESOLUTION = 30  # fixed time resolution for now
    
    rad, rad_vrs = ncf.load_nc(input_files[0])
    lid, lid_vrs = ncf.load_nc(input_files[1])
    mwr, mwr_vrs = ncf.load_nc(input_files[2])
    mod, mod_vrs = ncf.load_nc(input_files[3])

    try:
        freq = get_radar_freq(rad_vrs)
    except (ValueError, KeyError) as error:
        print(error)

    try:
        time = get_time(TIME_RESOLUTION)
    except ValueError as error:
        print(error)

           
    height = get_altitude_grid(rad_vrs['altitude'][:],
                               rad_vrs['range'][:])

    

def get_radar_freq(vrs):
    """ Return frequency of radar.

    Args:
        vrs: Pointer to radar variables.

    Returns:
        Frequency or radar.

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
    b1 = 30 < freq < 40
    b2 = 90 < freq < 100
    if not b1 and not b2:
        raise ValueError('Only 35 and 94 GHz radars supported.')
    return float(freq)


def get_altitude_grid(alt_radar, range_radar):
    """ Return altitude grid for Cloudnet products.
    Altitude grid is defined as the radar measurement 
    grid from the mean sea level.

    Args:
        alt_radar (float): Altitude of radar above mean sea level [km]
        range_radar (nd.array): Altitude grid of radar measurements 
                                above instrument [km]    

    Returns:
        (nd.array): Altitude grid

    """
    return range_radar + alt_radar


def get_time(reso):
    """ Computes fraction hour time vector 0-24 with user-given 
    resolution (in seconds) where 60 is the maximum allowed value.

    Args:
        reso (float): Time resolution in seconds.

    Returns:
        (nd.array): Time vector between 0 and 24.

    Raises:
        ValueError: Bad resolution as input.

    """
    if reso < 1 or reso > 60:
        raise ValueError('Time resolution should be between 0 and 60 [s]')
    step = reso/7200
    return np.arange(step, 24-step, step*2)
