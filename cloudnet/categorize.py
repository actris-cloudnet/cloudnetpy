""" This module is used to generate
categorize (Level 1) product from pre-processed
radar, lidar and MWR files.
"""

import sys
import math
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import config
import ncf
import utils
import atmos


def generate_categorize(input_files, output_file, aux):
    """ Generate Cloudnet Level 1 categorize file.

    Args:
        input_files (tuple): Tuple of strings containing full paths of
                             4 input files (radar, lidar, mwr, model).
        output_file (str): Full path of output file.
        aux (tuple): Tuple of strings including some metadata
                     of the site (site_name, institute).

    """
    try:
        time = utils.get_time(config.TIME_RESOLUTION)
        rad_vars, lid_vars, mwr_vars, mod_vars = _load_files(input_files)
        freq = ncf.get_radar_freq(rad_vars)
    except (ValueError, KeyError) as error:
        sys.exit(error)
    wlband = ncf.get_wl_band(freq)
    height = _get_altitude_grid(rad_vars)  # m
    vfold = rad_vars['NyquistVelocity'][:]
    try:
        alt_site = ncf.get_site_alt(rad_vars, lid_vars, mwr_vars)  # m
        radar = fetch_radar(rad_vars, ('Zh', 'v', 'ldr', 'width'), time, vfold)
    except KeyError as error:
        sys.exit(error)
    lidar = fetch_lidar(lid_vars, ('beta',), time, height)
    lwp = fetch_mwr(mwr_vars, config.LWP_ERROR, time)
    model = fetch_model(mod_vars, alt_site, wlband, time, height)


def _load_files(files):
    """ Wrapper to load input files (radar, lidar, mwr, model). """
    if len(files) != 4:
        raise ValueError('You need 4 input files: radar, lidar, mwr, model.')
    out = []
    for fil in files:
        out.append(ncf.load_nc(fil))
    return out


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


def fetch_radar(rad_vars, fields, time_new, vfold):
    """ Read and rebin radar 2d fields in time.

    Args:
        rad_vars: A netCDF instance.
        fields (tuple): Tuple of strings containing radar
                        fields to be averaged.
        time_new (array_like): A 1-D array.
        vfold (float): folding velocity

    Returns:
        (dict): Rebinned radar fields.

    Raises:
        KeyError: Missing field.

    Notes:
        Radar echo, 'Zh', is averaged in linear space.
        Doppler velocity, 'v', is averaged in polar coordinates.

    """
    out = {}
    c = math.pi/vfold
    time_orig = rad_vars['time'][:]
    for field in fields:
        if field not in rad_vars:
            raise KeyError(f"No variable '{field}' in the radar file.")
        data = rad_vars[field][:]
        if field == 'Zh':  # average in linear scale
            data_lin = utils.db2lin(data)
            data_mean = utils.rebin_2d(time_orig, data_lin, time_new)
            out[field] = utils.lin2db(data_mean)
        elif field == 'v':  # average in polar coordinates
            data = data * c
            vx, vy = np.cos(data), np.sin(data)
            vx_mean = utils.rebin_2d(time_orig, vx, time_new)
            vy_mean = utils.rebin_2d(time_orig, vy, time_new)
            out[field] = np.arctan2(vy_mean, vx_mean) / c
        else:
            out[field] = utils.rebin_2d(time_orig, data, time_new)
    return out


def fetch_lidar(lid_vars, fields, time, height):
    """ Read and rebin lidar 2d fields in time and height.

    Args:
        lid_vars: A netCDF instance.
        fields (tuple): Tuple of strings containing lidar
                        fields to be averaged.
        time (array_like): A 1-D array.
        height (array_like): A 1-D array.

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
        dataim = utils.rebin_2d(x, lid_vars[field][:], time)
        dataim = utils.rebin_2d(y, dataim.T, height).T
        out[field] = dataim
    return out


def fetch_mwr(mwr_vars, lwp_errors, time):
    """ Wrapper to read and interpolate LWP and its error.

    Args:
        mwr_vars: A netCDF instance.
        lwp_errors: A 2-element tuple containing
                    (fractional_error, linear_error)
        time (array_like): A 1-D array.

    Returns:
        Dict containing interpolated LWP data {'lwp', 'lwp_error'}

    """
    def _interpolate_lwp(time_lwp, lwp, time_new):
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
    lwp_i = _interpolate_lwp(lwp['time'], lwp['lwp'], time)
    lwp_error_i = _interpolate_lwp(lwp['time'], lwp['lwp_error'], time)
    return {'lwp': lwp_i, 'lwp_error': lwp_error_i}


def _read_lwp(mwr_vars, frac_err, lin_err):
    """ Read LWP, estimate its error, and convert time vector if needed.

    Args:
        mwr_vars: A netCDF4 instance.
        frac_error (float): Fractional error (scalar).
        lin_error (float): Linear error (scalar).

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


def fetch_model(mod_vars, alt_site, wlband, time, height):
    """ Wrapper function to read and interpolate model variables.

    Args:
        mod_vars: A netCDF4 instance.
        alt_site (int): Altitude of site above mean sea level.
        wlband (int): 0 (35Ghz) or 1 (94Ghz)
        time (array_like): A 1-D array.
        height (array_like): A 1-D array.

    Returns:
        Dict containing original model fields in common altitude
        grid, interpolated fields in Cloudnet time/height grid,
        and wet bulb temperature.

    """
    fields = ('temperature', 'pressure', 'rh', 'gas_atten',
              'specific_gas_atten', 'specific_saturated_gas_atten',
              'specific_liquid_atten')
    fields_all = fields + ('q', 'uwind', 'vwind')
    model, model_time, model_height = _read_model(mod_vars, fields_all,
                                                  alt_site, wlband)
    model_i = _interpolate_model(model, fields, model_time,
                                 model_height, time, height)
    Tw = atmos.wet_bulb(model_i['temperature'], model_i['pressure'],
                        model_i['rh'])
    return {'model': model, 'model_i': model_i, 'time': model_time,
            'height': model_height, 'Tw': Tw}


def _read_model(vrs, fields, alt_site, wlband):
    """Read model fields and interpolate into common altitude grid."""
    out = {}
    model_heights = ncf.km2m(vrs['height']) + alt_site  # above mean sea level
    model_heights = np.array(model_heights)  # masked arrays not supported
    model_time = vrs['time'][:]
    new_grid = np.mean(model_heights, axis=0)  # is mean profile ok?
    nx, ny = len(model_time), len(new_grid)
    for field in fields:
        data = np.array(vrs[field][:])
        datai = np.zeros((nx, ny))
        if 'atten' in field:
            data = data[wlband, :, :]
        # interpolate model profiles into common altitude grid
        for ind in range(nx):
            f = interp1d(model_heights[ind, :], data[ind, :],
                         fill_value='extrapolate')
            datai[ind, :] = f(new_grid)
        out[field] = datai
    return out, model_time, new_grid


def _interpolate_model(model, fields, *args):
    """ Interpolate model fields into Cloudnet time/height grid """
    out = {}
    for field in fields:
        out[field] = utils.interpolate_2d(*args, model[field])
    return out
