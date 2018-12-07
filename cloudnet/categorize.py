""" This module is used to generate
categorize (Level 1) product from pre-processed
radar, lidar and MWR files.
"""

import numpy as np
import numpy.ma as ma
from scipy import stats
import ncf
import utils

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
        freq = ncf.get_radar_freq(rad_vrs)
    except (ValueError, KeyError) as error:
        print(error)

    try:
        time = utils.get_time(TIME_RESOLUTION)
    except ValueError as error:
        print(error)

    height = get_altitude_grid(rad_vrs['altitude'][:],
                               rad_vrs['range'][:])

    site_altitude = get_site_altitude(rad_vrs['altitude'][:],
                                      lid_vrs['altitude'][:])

    # average radar variables in time
    fields = ('Zh', 'v', 'ldr', 'width')
    try:
        radar = fetch_radar(rad_vrs, fields, time)
    except KeyError as error:
        print(error)

    vfold = rad_vrs['NyquistVelocity'][:]

    # average lidar variables in time and height
    lidar = fetch_lidar(lid_vrs, ('beta',), time, height)
    

def get_site_altitude(alt_radar, alt_lidar):
    """ Return altitude of the measurement site above mean sea level.

    Site altitude is the altitude of radar or lidar, which one is lower.

    Args:
        alt_radar (float): Altitude of radar above mean sea level [km]
        alt_lidar (float): Altitude of lidar above mean sea level [km]

    Returns:
        Altitude of the measurement site.

    """
    return min(alt_radar, alt_lidar)


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


def fetch_radar(vrs, fields, time):
    """ Read and rebin radar 2d fields in time.

    Args:
        vrs: Pointer to radar variables
        fields (tuple): Tuple of strings containing radar
                        fields to be averaged.
        time: New time vector.

    Returns:
        (dict): Rebinned radar fields.

    Raises:
        KeyError: Missing field.

    """
    out = {}
    x = vrs['time'][:]
    for field in fields:
        if field not in vrs:
            raise KeyError(f"No variable '{field}' in the radar file.")
        out[field] = utils.rebin_x_2d(x, vrs[field][:], time)
    return out


def fetch_lidar(vrs, fields, time, height):
    """ Read and rebin lidar 2d fields in time and height.

    Args:
        vrs: Pointer to lidar variables
        fields (tuple): Tuple of strings containing lidar
                        fields to be averaged.
        time: New time vector.

    Returns:
        (dict): Rebinned lidar fields.

    Raises:
        KeyError: Missing field.

    """
    out = {}        
    x = vrs['time'][:]
    lidar_alt = ncf.km2m(vrs['altitude'])
    y = ncf.km2m(vrs['range']) + lidar_alt
    for field in fields:
        if field not in vrs:
            raise KeyError(f"No variable '{field}' in the lidar file.")
        dataim = utils.rebin_x_2d(x, vrs[field][:], time)
        dataim = utils.rebin_x_2d(y, dataim.T, height).T
        out[field] = dataim
    return out




    
