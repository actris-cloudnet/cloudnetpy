""" This module is used to generate
categorize (Level 1) product from pre-processed
radar, lidar and MWR files.
"""

import os
import sys
sys.path.insert(0, os.path.abspath('../../cloudnetpy'))  #pylint: disable=wrong-import-position
import math
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
from collections import namedtuple
# from datetime import datetime, timezone
from cloudnetpy import config
from cloudnetpy import ncf
from cloudnetpy import utils
from cloudnetpy import atmos
from cloudnetpy import classify
from cloudnetpy import output
from cloudnetpy.output import CnetVar
# from cloudnetpy import plotting


def generate_categorize(input_files, output_file, aux):
    """Generates Cloudnet Level 1 categorize file.

    Args:
        input_files (tuple): Tuple of strings containing full paths of
                             4 input files (radar, lidar, mwr, model).
        output_file (str): Full path of output file.
        aux (tuple): Tuple of strings including some metadata
                     of the site (site_name, institute).

    References:
        https://journals.ametsoc.org/doi/10.1175/BAMS-88-6-883

    """
    try:
        time = utils.get_time(config.TIME_RESOLUTION)
        rad_vars, lid_vars, mwr_vars, mod_vars = _load_files(input_files)
        radar_type, dvec = ncf.fetch_radar_meta(input_files[0])
        freq = ncf.get_radar_freq(rad_vars)
    except (ValueError, KeyError) as error:
        sys.exit(error)
    height = _get_altitude_grid(rad_vars)  # m
    try:
        alt_site = ncf.get_site_alt(rad_vars, lid_vars, mwr_vars)  # m
        radar = fetch_radar(rad_vars, ('Zh', 'v', 'ldr', 'width'), time)
    except KeyError as error:
        sys.exit(error)
    lidar = fetch_lidar(lid_vars, ('beta',), time, height)
    lwp = fetch_mwr(mwr_vars, config.LWP_ERROR, time)
    model = fetch_model(mod_vars, alt_site, freq, time, height)
    bits = classify.fetch_cat_bits(radar, lidar['beta'], model['Tw'], time, height)
    atten = _get_attenuations(lwp, model['interp'], bits, height)
    qual_bits = classify.fetch_qual_bits(radar['Zh'], lidar['beta'],
                                         bits['clutter_bit'], atten['liq_atten'])
    Z_corr = _correct_atten(radar['Zh'], atten['gas_atten'],
                            atten['liq_atten']['liq_atten'])
    Z_err = _fetch_Z_errors(radar, rad_vars, atten, bits['clutter_bit'], time, freq)
    # Collect variables for output file writing:
    cat_vars = {'height': height,
                'time': time,
                'latitude': rad_vars['latitude'][:],
                'longitude': rad_vars['longitude'][:],
                'altitude': alt_site,
                'radar_frequency': freq,
                'lidar_wavelength': lid_vars['wavelength'][:],
                'beta': lidar['beta'],
                'beta_error': config.BETA_ERROR,
                'beta_bias': config.BETA_BIAS,
                'Z': Z_corr,
                'v': radar['v'],
                'width': radar['width'],
                'ldr': radar['ldr'],
                'Z_bias': config.Z_BIAS,
                'temperature': model['original']['temperature'],
                'pressure': model['original']['pressure'],
                'specific_humidity': model['original']['q'],
                'uwind': model['original']['uwind'],
                'vwind': model['original']['vwind'],
                'model_height': model['height'],
                'model_time': model['time'],
                'category_bits': bits['cat_bits'],
                'Tw': model['Tw'],
                'insect_probability': bits['insect_prob'],
                'radar_gas_atten': atten['gas_atten'],
                'radar_liquid_atten': atten['liq_atten']['liq_atten'],
                'lwp': lwp['lwp'],
                'lwp_error': lwp['lwp_error'],
                'quality_bits': qual_bits,
                'Z_error': Z_err['error'],
                'Z_sensitivity': Z_err['sensitivity']}
    obs = _cat_cnet_vars(cat_vars, dvec, radar_type)
    output.save_cat(output_file, time, height, model['time'], model['height'],
                    obs, dvec, aux)


def _correct_atten(Z, gas_atten, liq_atten):
    """Corrects radar echo for attenuation.

    Args:
        Z (MaskedArray): Radar echo.
        gas_atten (ndarray): attenuation due to atmospheric gases.
        liq_atten (MaskedArray): attenuation due to atmospheric liquid.

    Returns:
        Copy of input Z, corrected by liquid attenuation
        (where applicable) and gas attenuation.

    """
    Z_corr = ma.copy(Z) + gas_atten
    ind = ~liq_atten.mask
    Z_corr[ind] = Z_corr[ind] + liq_atten[ind]
    return Z_corr


def _get_attenuations(lwp, model_i, bits, height):
    """Returns attenuations due to atmospheric liquid and gases."""
    gas_atten = atmos.get_gas_atten(model_i, bits['cat_bits'], height)
    liq_atten = atmos.get_liquid_atten(lwp, model_i, bits, height)
    return {'gas_atten': gas_atten, 'liq_atten': liq_atten}


def _load_files(files):
    """Wrapper to load input files (radar, lidar, mwr, model). """
    if len(files) != 4:
        raise ValueError('Aborting - there should be excatly 4 input files.')
    out = []
    for fil in files:
        out.append(ncf.load_nc(fil))
    return out


def _get_altitude_grid(rad_vars):
    """Returns altitude grid for Cloudnet products (m).
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
    """Reads and rebins radar 2d fields in time.

    Args:
        rad_vars: A netCDF instance.
        fields (tuple): Tuple of strings containing radar
                        fields to be averaged.
        time_new (ndarray): A 1-D array.

    Returns:
        (dict): Rebinned radar fields.

    Raises:
        KeyError: Missing field.

    Notes:
        Radar echo, 'Zh', is averaged in linear space.
        Doppler velocity, 'v', is averaged in polar coordinates.

    """
    out = {}
    vfold = math.pi/rad_vars['NyquistVelocity'][:]
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
            data = data * vfold
            vx, vy = np.cos(data), np.sin(data)
            vx_mean = utils.rebin_2d(time_orig, vx, time_new)
            vy_mean = utils.rebin_2d(time_orig, vy, time_new)
            out[field] = np.arctan2(vy_mean, vx_mean) / vfold
        else:
            out[field] = utils.rebin_2d(time_orig, data, time_new)
    return out


def fetch_lidar(lid_vars, fields, time, height):
    """Reads and rebins lidar 2d fields in time and height.

    Args:
        lid_vars: A netCDF instance.
        fields (tuple): Tuple of strings containing lidar
                        fields to be averaged.
        time (ndarray): A 1-D array.
        height (ndarray): A 1-D array.

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
    """Wrapper to read and interpolate LWP and its error.

    Args:
        mwr_vars: A netCDF instance.
        lwp_errors: A 2-element tuple containing
                    (fractional_error, linear_error)
        time (ndarray): A 1-D array.

    Returns:
        Dict containing interpolated LWP data {'lwp', 'lwp_error'}

    """
    def _interpolate_lwp(time_lwp, lwp, time_new):
        """Linear interpolation of LWP data. This can be
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
    """Reads LWP, estimates its error, and converts time if needed.

    Args:
        mwr_vars: A netCDF4 instance.
        frac_error (float): Fractional error (scalar).
        lin_error (float): Linear error (scalar).

    Returns:
        Dict containing {'time', 'lwp', 'lwp_error'} that are 1-D arrays.


    Note:
        hatpro time can be 'hours since' 00h of measurement date
        or 'seconds since' some epoch (which could be site/file
        dependent).

    """
    lwp = mwr_vars['LWP_data'][:]
    time = mwr_vars['time'][:]
    if max(time) > 24:
        time = utils.epoch2desimal_hour((2001, 1, 1), time)  # fixed epoc!!
    lwp_err = np.sqrt(lin_err**2 + (frac_err*lwp)**2)
    return {'time': time, 'lwp': lwp, 'lwp_error': lwp_err}


def fetch_model(mod_vars, alt_site, freq, time, height):
    """ Wrapper function to read and interpolate model variables.

    Args:
        mod_vars: A netCDF4 instance.
        alt_site (int): Altitude of site above mean sea level.
        freq (float): Radar frequency.
        time (ndarray): A 1-D array.
        height (ndarray): A 1-D array.

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
                                                  alt_site, freq)
    model_i = _interpolate_model(model, fields, model_time,
                                 model_height, time, height)
    Tw = atmos.wet_bulb(model_i['temperature'], model_i['pressure'],
                        model_i['rh'])
    return {'original': model, 'interp': model_i, 'time': model_time,
            'height': model_height, 'Tw': Tw}


def _read_model(vrs, fields, alt_site, freq):
    """Read model fields and interpolate into common altitude grid.

    Args:
        vrs: A netCDF4 instance.
        fields (array_like): list of strings containing fields
            to be interpolated.
        alt_site (float): Site altitude (m).
        freq (float): Radar frequency (GHz).

    Returns:
        3-element tuple containing (1) dict that has original model fields
        in common altitude grid, and interpolated model fields, (2) Original
        model time (3) Original model heights (average of individual heights
        of the day).
    """
    out = {}
    wlband = ncf.get_wl_band(freq)
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
    """ Interpolate model fields into Cloudnet time/height grid

    Args:
        model: A netCDF instance.
        fields (array_like): list of strings containing fields
            to be interpolated.
        *args: original time, orignal height, new time, new height.

    Returns:
        dict containing interpolated model fields.

    """
    out = {}
    for field in fields:
        out[field] = utils.interpolate_2d(*args, model[field])
    return out


def _fetch_Z_errors(radar, rad_vars, atten, clutter_bit, time, freq):
    """Returns sensitivity, precision and error of radar echo.

    Args:
        radar: A netCDF4 instance.
        rad_vars: Radar variables.
        atten (dict): Gas and liquid attenuation variables.
        clutter_bit (ndarray): Boolean array denoting pixels
            contaminated by clutter.
        time (ndarray): Time vector.
        freq (float): Radar frequency.

    Returns:
        Dict containing {'radar_sensitivity', 'radar_error'}.

    Notes:
        Needs to be at least checked and perhaps refactored.

    """
    Z = radar['Zh']
    gas_atten, liq_atten = atten['gas_atten'], atten['liq_atten']
    radar_range = ncf.km2m(rad_vars['range'])
    log_range = utils.lin2db(radar_range, scale=20)
    Z_power = Z - log_range
    Z_power_list = np.sort(Z_power.compressed())
    Z_power_min = Z_power_list[int(np.floor(len(Z_power_list)/1000))]
    # Sensitivity:
    Z_sensitivity = Z_power_min + log_range + np.mean(gas_atten, axis=0)
    Zc = ma.masked_where(clutter_bit == 0, Z, copy=True)
    Zc = ma.median(Zc, axis=0)
    ind = ~Zc.mask
    Z_sensitivity[ind] = Zc[ind]
    # Precision:
    dwell_time = (time[1]-time[0])*3600
    independent_pulses = (dwell_time*4*np.sqrt(math.pi)*freq*1e9/3e8)*radar['width']
    Z_precision = 4.343*(1.0/np.sqrt(independent_pulses) +
                         utils.db2lin(Z_power_min-Z_power)/3)
    # Error:
    g_prec = config.GAS_ATTEN_PREC
    Z_error = utils.l2norm(gas_atten*g_prec, liq_atten['liq_atten_err'], Z_precision)
    Z_error[liq_atten['liq_atten_ucorr_bit'] == 1] = None
    return {'sensitivity': Z_sensitivity, 'error': Z_error}


def _anc_names(var, bias=False, err=False, sens=False):
    """Returns list of ancillary variable names."""
    out = ''
    if bias:
        out = out + var + '_bias '
    if err:
        out = out + var + '_error '
    if sens:
        out = out + var + '_sensitivity '
    return out[:-1]


def _cat_cnet_vars(vars_in, dvec, radar_type):
    """Creates list of variable instances for output writing."""
    lin, log = 'linear', 'logarithmic'
    src = 'source'
    anc = 'ancillary_variables'
    bias_comm = 'This variable is an estimate of the one-standard-deviation calibration error'
    model_source = 'HYSPLIT'
    radar_source = f"{radar_type} cloud radar"
    obs = []

    # general variables
    var = 'height'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Height above mean sea level',
                       size=('height'),
                       units='m',
                       fill_value=None))
    var = 'time'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Time UTC',
                       size=('time'),
                       units='hours since ' + dvec + ' 00:00:00 +0:00',
                       fill_value=None,
                       comment='Fixed ' + str(config.TIME_RESOLUTION) + 's resolution.'))
    var = 'model_height'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Height of model variables above mean sea level',
                       size=('model_height'),
                       units='m',
                       fill_value=None))
    var = 'model_time'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Model time UTC',
                       size=('model_time'),
                       units='hours since ' + dvec + ' 00:00:00 +0:00',
                       fill_value=None))
    var = 'latitude'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Latitude of site',
                       size=(),
                       units='degrees_north',
                       fill_value=None))
    var = 'longitude'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Longitude of site',
                       size=(),
                       units='degrees_east',
                       fill_value=None))
    var = 'altitude'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Altitude of site',
                       size=(),
                       units='m',
                       fill_value=None,
                       comment=_comments(var)))
    # radar variables
    var = 'radar_frequency'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Transmit frequency',
                       size=(),
                       units='GHz',
                       fill_value=None))
    var = 'Z'
    lname = 'Radar reflectivity factor'
    obs.append(CnetVar(var, vars_in[var],
                       long_name=lname,
                       units='dBZ',
                       plot_range=(-40, 20),
                       plot_scale=lin,
                       extra_attributes={src: radar_source,
                                         anc: _anc_names(var, True, True, True)}))
    var = 'Z_bias'
    obs.append(CnetVar(var, vars_in[var],
                       long_name=output.bias_name(lname),
                       size=(),
                       units='dB',
                       fill_value=None,
                       comment=bias_comm))
    var = 'Z_error'
    obs.append(CnetVar(var, vars_in[var],
                       long_name=output.err_name(lname),
                       units='dB'))
    var = 'Z_sensitivity'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Minimum detectable radar reflectivity',
                       size=('height'),
                       units='dBZ',
                       comment=_comments(var)))
    var = 'v'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Doppler velocity',
                       units='m s-1',
                       plot_range=(-4, 2),
                       plot_scale=lin,
                       comment=_comments(var),
                       extra_attributes={src:radar_source}))
    var = 'width'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Spectral width',
                       units='m s-1',
                       plot_range=(0.03, 3),
                       plot_scale=log,
                       extra_attributes={src:radar_source}))
    var = 'ldr'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Linear depolarisation ratio',
                       units='dB',
                       plot_range=(-30, 0),
                       plot_scale=lin,
                       extra_attributes={src:radar_source}))
    # lidar variables
    var = 'lidar_wavelength'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Laser wavelength',
                       size=(),
                       units='nm',
                       fill_value=None))
    var = 'beta'
    lname = 'Attenuated backscatter coefficient'
    obs.append(CnetVar(var, vars_in[var],
                       long_name=lname,
                       units='sr-1 m-1',
                       plot_range=(1e-7, 1e-4),
                       plot_scale=log,
                       extra_attributes={src:'Lidar/Ceilometer model XXX',
                                         anc: _anc_names(var, bias=True, err=True)}))
    var = 'beta_bias'
    obs.append(CnetVar(var, vars_in[var],
                       long_name=output.bias_name(lname),
                       size=(),
                       units='dB',
                       fill_value=None,
                       comment=bias_comm))
    var = 'beta_error'
    obs.append(CnetVar(var, vars_in[var],
                       long_name=output.err_name(lname),
                       size=(),
                       units='dB',
                       fill_value=None))
    # mwr variables
    var = 'lwp'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Liquid water path',
                       size=('time'),
                       units='g m-2',
                       plot_range=(-100, 1000),
                       plot_scale=lin,
                       extra_attributes={'source':'HATPRO microwave radiometer'}))
    var = 'lwp_error'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Error in liquid water path, one standard deviation',
                       size=('time'),
                       units='g m-2'))
    # model variables
    var = 'temperature'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Temperature',
                       size=('model_time', 'model_height'),
                       units='K',
                       plot_range=(200, 300),
                       plot_scale=lin,
                       extra_attributes={src:model_source}))
    var = 'pressure'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Pressure',
                       size=('model_time', 'model_height'),
                       units='Pa',
                       plot_range=(0, 1.1e5),
                       plot_scale=log,
                       extra_attributes={src:model_source}))
    var = 'specific_humidity'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Model specific humidity',
                       size=('model_time', 'model_height'),
                       plot_range=(0, 0.006),
                       plot_scale=lin,
                       extra_attributes={src:model_source}))
    var = 'uwind'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Zonal wind',
                       size=('model_time', 'model_height'),
                       units='m s-1',
                       plot_range=(-50, 50),
                       plot_scale=lin,
                       extra_attributes={src:model_source}))
    var = 'vwind'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Meridional wind',
                       size=('model_time', 'model_height'),
                       units='m s-1',
                       plot_range=(-50, 50),
                       plot_scale=lin,
                       extra_attributes={src:model_source}))
    # other
    var = 'category_bits'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Target classification bits',
                       data_type='i4',
                       fill_value=None,
                       extra_attributes={'valid_range': [0, 5],
                                         'flag_masks': [0, 1, 2, 3, 4, 5],
                                         'flag_meanings':('liquid_droplets falling_hydrometeors '
                                                          'freezing_temperature melting_ice '
                                                          'aerosols insects')}))
    var = 'quality_bits'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Data quality bits',
                       data_type='i4',
                       fill_value=None,
                       extra_attributes={'valid_ranges': [0, 5],
                                         'flag_masks': [0, 1, 2, 3, 4, 5],
                                         'flag_meanings':('lidar_echo radar_echo radar_clutter '
                                                          'lidar_molec_scatter attenuation '
                                                          'atten_correction')}))
    var = 'Tw'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Wet bulb temperature',
                       units='K',
                       fill_value=None,
                       comment=_comments(var)))
    var = 'insect_probability'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Probability of insects',
                       fill_value=None))
    var = 'radar_gas_atten'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Two-way radar attenuation due to atmospheric gases',
                       units='dB',
                       plot_range=(0, 4),
                       plot_scale=lin,
                       comment=_comments(var)))
    var = 'radar_liquid_atten'
    obs.append(CnetVar(var, vars_in[var],
                       long_name='Approximate two-way radar attenuation due to liquid water',
                       units='dB',
                       plot_range=(0, 4),
                       plot_scale=lin,
                       comment=_comments(var)))
    return obs


def _comments(field):
    """Returns the comment text for a Cloudnet variable."""
    com = {
        'radar_liquid_atten': ('This variable was calculated from the liquid water path\n'
                               'measured by microwave radiometer using lidar and radar returns to perform\n'
                               'an approximate partioning of the liquid water content with height. Bit 5 of the\n'
                               'quality_bits variable indicates where a correction for liquid water attenuation has\n'
                               'been performed.'),
        
        'radar_gas_atten': ('This variable was calculated from the model temperature,\n'
                            'pressure and humidity, but forcing pixels containing liquid cloud to saturation\n'
                            'with respect to liquid water. It was calculated using the millimeter-wave propagation\n'
                            'model of Liebe (1985, Radio Sci. 20(5), 1069-1089). It has been used to correct Z.'),
        
        'Tw': ('Calculated from model T, P and relative humidity,\n'
               'which were first interpolated into measurement grid.'),

        'v': ('This parameter is the radial component of the velocity,\n'
              'with positive velocities are away from the radar.'),
        
        'Z_sensitivity': ('This variable is an estimate of the radar sensitivity,\n'
                          'i.e. the minimum detectable radar reflectivity, as a function of height.\n'
                          'It includes the effect of ground clutter and gas attenuation but not liquid attenuation.'),

        'altitude': ('Defined as the altitude of radar or lidar, '
                     'choosing the one that is lower.')
    }
    return com[field]

