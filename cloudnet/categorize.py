""" This module is used to generate
categorize (Level 1) product from pre-processed
radar, lidar and MWR files.
"""

import sys
import numpy as np
import numpy.ma as ma
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

    TIME_RESOLUTION = 60  # fixed time resolution for now

    rad_vars = ncf.load_nc(input_files[0])
    lid_vars = ncf.load_nc(input_files[1])
    mwr_vars = ncf.load_nc(input_files[2])
    mod_vars = ncf.load_nc(input_files[3])

    try:
        freq = ncf.get_radar_freq(rad_vars)
    except (ValueError, KeyError) as error:
        print(error)

    try:
        time = utils.get_time(TIME_RESOLUTION)
    except ValueError as error:
        print(error)

    height = get_altitude_grid(rad_vars)

    try:
        site_alt = ncf.get_site_alt(rad_vars, lid_vars)
    except KeyError as error:
        print(error)
        
    # average radar variables in time
    fields = ('Zh', 'v', 'ldr', 'width')
    try:
        radar = fetch_radar(rad_vars, fields, time)
    except KeyError as error:
        print(error)
    vfold = rad_vars['NyquistVelocity'][:]

    # average lidar variables in time and height
    lidar = fetch_lidar(lid_vars, ('beta',), time, height)


def get_altitude_grid(rad_vars):
    """ Return altitude grid for Cloudnet products.
    Altitude grid is defined as the radar measurement
    grid from the mean sea level.

    Args:
        rad_vars: Radar variables.

    Returns:
        Altitude grid

    """
    return rad_vars['range'][:] + rad_vars['altitude']


def fetch_radar(rad_vars, fields, time_new):
    """ Read and rebin radar 2d fields in time.

    Args:
        rad_vars: Pointer to radar variables
        fields (tuple): Tuple of strings containing radar
                        fields to be averaged.
        time_new: A 1-D array.

    Returns:
        (dict): Rebinned radar fields.

    Raises:
        KeyError: Missing field.

    """
    out = {}
    time_orig = rad_vars['time'][:]
    for field in fields:
        if field not in rad_vars:
            raise KeyError(f"No variable '{field}' in the radar file.")
        out[field] = utils.rebin_x_2d(time_orig, rad_vars[field][:], time_new)
    return out


def fetch_lidar(lid_vars, fields, time, height):
    """ Read and rebin lidar 2d fields in time and height.

    Args:
        lid_vars: Pointer to lidar variables
        fields (tuple): Tuple of strings containing lidar
                        fields to be averaged.
        time: A 1-D array.
        height: A 1-D array.

    Returns:
        (dict): Rebinned lidar fields.

    Raises:
        KeyError: Missing field.

    """
    out = {}
    x = lid_vars['time'][:]
    lidar_alt = ncf.km2m(lid_vars['altitude'])
    y = ncf.km2m(lid_vars['range']) + lidar_alt
    for field in fields:
        if field not in lid_vars:
            raise KeyError(f"No variable '{field}' in the lidar file.")
        dataim = utils.rebin_x_2d(x, lid_vars[field][:], time)
        dataim = utils.rebin_x_2d(y, dataim.T, height).T
        out[field] = dataim
    return out
