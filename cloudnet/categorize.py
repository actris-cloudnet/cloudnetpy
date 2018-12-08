""" This module is used to generate
categorize (Level 1) product from pre-processed
radar, lidar and MWR files.
"""

import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
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
    LWP_ERROR = (0.25, 20)  # fractional and linear error components

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

    height = _get_altitude_grid(rad_vars)  # m

    try:
        site_alt = ncf.get_site_alt(rad_vars, lid_vars)  # m
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

    # interpolate mwr variables in time
    lwp = fetch_mwr(mwr_vars, LWP_ERROR, time)


def _get_altitude_grid(rad_vars):
    """ Return altitude grid for Cloudnet products in [m].
    Altitude grid is defined as the instruments measurement
    grid from the mean sea level.

    Args:
        rad_vars: A netCDF4 instance.

    Returns:
        Altitude grid.

    Notes:
        Grid should be calculated from radar measurement.

    """
    range_instru = ncf.km2m(rad_vars['range'])
    alt_instru = ncf.km2m(rad_vars['altitude'])
    return range_instru + alt_instru


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
    y = ncf.km2m(lid_vars['range']) + lidar_alt  # m
    for field in fields:
        if field not in lid_vars:
            raise KeyError(f"No variable '{field}' in the lidar file.")
        dataim = utils.rebin_x_2d(x, lid_vars[field][:], time)
        dataim = utils.rebin_x_2d(y, dataim.T, height).T
        out[field] = dataim
    return out


def fetch_mwr(mwr_vars, lwp_errors, time):
    """ Wrapper to read and interpolate LWP and its error.

    Args:
        mwr_vars: A netCDF instance.
        lwp_errors: A 2-element tuple containing
                    (fractional_error, linear_error)
        time: A 1-D array.

    Returns:
        Dict containing interpolated LWP data {'lwp', 'lwp_error'}

    """
    def interpolate_lwp(time_lwp, lwp, time_new):
        """ Linear interpolation of LWP data. This can be
        bad idea if there are lots of gaps in the data.
        """
        try:
            f = interp1d(time_lwp, lwp)
            lwp_i = f(time_new)
        except:
            lwp_i = np.full_like(time_new, fill_value=np.nan)
        return lwp_i

    lwp = _read_lwp(mwr_vars, *lwp_errors)
    lwp_i = interpolate_lwp(lwp['time'], lwp['lwp'], time)
    lwp_error_i = interpolate_lwp(lwp['time'], lwp['lwp_error'], time)
    return {'lwp': lwp_i, 'lwp_error': lwp_error_i}


def _read_lwp(mwr_vars, frac_err, lin_err):
    """ Read LWP, estimate its error, and convert time vector if needed.

    Args:
        mwr_vars: A netCDF4 instance.
        frac_error: Fractional error (scalar).
        lin_error: Linear error (scalar).

    Returns:
        Dict containing {'time', 'lwp', 'lwp_error'} that are 1-D arrays.


    Note: hatpro time can be 'hours since' 00h of measurement date
    or 'seconds since' some epoch (which could be site/file dependent).

    """
    lwp = mwr_vars['LWP_data'][:]
    time = mwr_vars['time'][:]
    if max(time) > 24:
        time = utils.epoch2desimal_hour((2001, 1, 1), time)  # fixed epoc!!
    lwp_err = np.sqrt(lin_err**2 + (frac_err*lwp)**2)
    return {'time': time, 'lwp': lwp, 'lwp_error': lwp_err}
