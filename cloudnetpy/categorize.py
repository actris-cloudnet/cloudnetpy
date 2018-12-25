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
from cloudnetpy import config
from cloudnetpy import ncf
from cloudnetpy import utils
from cloudnetpy import atmos
from cloudnetpy import classify
from cloudnetpy import output
from cloudnetpy.output import CnetVar as CV
# from cloudnetpy import plotting
# import matplotlib as mpl
# import matplotlib.pyplot as plt


def generate_categorize(input_files, output_file):
    """Generates Cloudnet Level 1 categorize file.

    Args:
        input_files (tuple): Tuple of strings containing full paths of
                             the 4 input files (radar, lidar, mwr, model).
        output_file (str): Full path of the output file.

    References:
        https://journals.ametsoc.org/doi/10.1175/BAMS-88-6-883

    """
    try:
        time = utils.get_time()
        rad_vars, lid_vars, mwr_vars, mod_vars = _load_files(input_files)
        radar_meta = ncf.fetch_radar_meta(input_files[0])
        height = _get_altitude_grid(rad_vars)  # m
    except (ValueError, KeyError) as error:
        sys.exit(error)
    try:
        alt_site = ncf.get_site_alt(rad_vars, lid_vars, mwr_vars)  # m
        radar = fetch_radar(rad_vars, ('Zh', 'v', 'ldr', 'width'), time)
    except KeyError as error:
        sys.exit(error)
    lidar = fetch_lidar(lid_vars, ('beta',), time, height)
    lwp = fetch_mwr(mwr_vars, config.LWP_ERROR, time)
    model = fetch_model(mod_vars, alt_site, radar_meta['freq'], time, height)
    bits = classify.fetch_cat_bits(radar, lidar['beta'], model['Tw'], time, height)
    gas_atten, liq_atten = _get_attenuations(lwp, model['interp'], bits, height)
    qual_bits = classify.fetch_qual_bits(radar['Zh'], lidar['beta'],
                                         bits['clutter'], liq_atten)
    Z_corr = _correct_atten(radar['Zh'], gas_atten, liq_atten['value'])
    Z_err = _fetch_Z_errors(radar, rad_vars, gas_atten, liq_atten,
                            bits['clutter'], time, radar_meta['freq'])
    instruments = ncf.fetch_instrument_models(*input_files[0:3])
    # Collect variables for output file writing:
    cat_vars = {'height': height,
                'time': time,
                'latitude': rad_vars['latitude'][:],
                'longitude': rad_vars['longitude'][:],
                'altitude': alt_site,
                'radar_frequency': radar_meta['freq'],
                'lidar_wavelength': lid_vars['wavelength'][:],
                'beta': lidar['beta'],
                'beta_error': config.BETA_ERROR[0],
                'beta_bias': config.BETA_ERROR[1],
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
                'category_bits': bits['cat'],
                'Tw': model['Tw'],
                'insect_probability': bits['insect_prob'],
                'radar_gas_atten': gas_atten,
                'radar_liquid_atten': liq_atten['value'],
                'lwp': lwp['value'],
                'lwp_error': lwp['err'],
                'quality_bits': qual_bits,
                'Z_error': Z_err['error'],
                'Z_sensitivity': Z_err['sensitivity']}
    obs = _cat_cnet_vars(cat_vars, radar_meta, instruments)
    output.save_cat(output_file, time, height, model['time'], model['height'],
                    obs, radar_meta)


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
    """Returns attenuations due to atmospheric liquid and gases.

    Args:
        lwp (dict): LWP variables {'value', 'err'}.
        model_i (dict): Interpolated model fields.
        bits (dict): Categorize bits {'cat', 'rain'}

    Returns:
        A 2-element tuple containing gas_atten (ndarray) and
        liq_atten {dict}.

    """
    gas_atten = atmos.get_gas_atten(model_i, bits['cat'], height)
    liq_atten = atmos.get_liquid_atten(lwp, model_i, bits, height)
    return gas_atten, liq_atten


def _load_files(files):
    """Wrapper to load input files (radar, lidar, mwr, model). 
    
    Args:
        files (tuple): Tuple of strings containing full paths of
            the 4 input netCDF files (radar, lidar, mwr, model).

    Returns: 
        (list): List of 4 elements containing the corresponding 
            netCDF variable dictionaries.

    Raises:
        ValueError: If there are not exactly 4 input files.

    """
    if len(files) != 4:
        raise ValueError('Aborting - there should be excatly 4 input files.')
    out = []
    for f in files:
        out.append(ncf.load_nc(f))
    return out


def _get_altitude_grid(rad_vars):
    """Returns altitude grid for Cloudnet products (m).
    Altitude grid is defined as the instruments measurement
    grid from the mean sea level.

    Args:
        rad_vars: A netCDF4 instance.

    Returns:
        Altitude grid.

    Raises:
        ValueError: Masked values in radar altitude. Not really the
        correct error type.

    """
    range_instru = ncf.km2m(rad_vars['range'])
    if ma.is_masked(range_instru):
        raise ValueError('Masked altitude values in radar data!?')
    alt_instru = ncf.km2m(rad_vars['altitude'])
    return np.array(range_instru + alt_instru)


def fetch_radar(rad_vars, fields, time_new):
    """Reads and rebins radar 2d fields in time.

    Args:
        rad_vars: A netCDF instance.
        fields (tuple): Tuple of strings containing radar
                        fields to be averaged.
        time_new (ndarray): A 1-D array.

    Returns:
        Dict containing rebinned radar fields.

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
        Dict containing rebinned lidar fields.


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
        Dict containing interpolated LWP data
        and its error {'value', 'err'}.

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
    lwp_i = _interpolate_lwp(lwp['time'], lwp['value'], time)
    lwp_error_i = _interpolate_lwp(lwp['time'], lwp['err'], time)
    return {'value': lwp_i, 'err': lwp_error_i}


def _read_lwp(mwr_vars, frac_err, lin_err):
    """Reads LWP, estimates its error, and converts time if needed.

    Args:
        mwr_vars: A netCDF4 instance.
        frac_error (float): Fractional error (scalar).
        lin_error (float): Linear error (scalar).

    Returns:
        Dict containing {'time', 'value', 'err'} that are 1-D arrays.

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
    return {'time': time, 'value': lwp, 'err': lwp_err}


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
        Dict containing interpolated model fields.

    """
    out = {}
    for field in fields:
        out[field] = utils.interpolate_2d(*args, model[field])
    return out


def _fetch_Z_errors(radar, rad_vars, gas_atten, liq_atten,
                    clutter_bit, time, freq):
    """Returns sensitivity, precision and error of radar echo.

    Args:
        radar: A netCDF4 instance.
        rad_vars: Radar variables.
        gas_atten (ndarray): Gas attenuation.
        liq_atten (dict): Liquid attenuation,
            containing {'err', 'ucorr_bit'}
        clutter_bit (ndarray): Boolean array denoting pixels
            contaminated by clutter.
        time (ndarray): Time vector.
        freq (float): Radar frequency.

    Returns:
        Dict containing {'Z_sensitivity', 'Z_error'} which are
        1D and 2D MaskedArrays, respectively.

    Notes:
        Needs to be at least checked and perhaps refactored.

    """
    Z = radar['Zh']
    radar_range = ncf.km2m(rad_vars['range'])
    log_range = utils.lin2db(radar_range, scale=20)
    Z_power = Z - log_range
    Z_power_list = np.sort(Z_power.compressed())
    Z_power_min = Z_power_list[int(np.floor(len(Z_power_list)/1000))]
    # Sensitivity:
    Z_sensitivity = Z_power_min + log_range + np.mean(gas_atten, axis=0)
    Zc = ma.masked_where(~clutter_bit, Z)
    Zc = ma.median(Zc, axis=0)
    ind = ~Zc.mask
    Z_sensitivity[ind] = Zc[ind]
    # Precision:
    dwell_time = (time[1]-time[0])*3600
    independent_pulses = (dwell_time*4*np.sqrt(math.pi)*freq*1e9/3e8)*radar['width']
    Z_precision = 4.343*(1.0/np.sqrt(independent_pulses) +
                         utils.db2lin(Z_power_min-Z_power)/3)
    # Error:
    Z_error = utils.l2norm(gas_atten*config.GAS_ATTEN_PREC, liq_atten['err'],
                           Z_precision)
    Z_error = ma.masked_where(liq_atten['ucorr_bit'], Z_error)
    return {'sensitivity': Z_sensitivity, 'error': Z_error}


def _anc_names(var, bias=False, err=False, sens=False):
    """Returns list of ancillary variable names."""
    out = ''
    if bias:
        out += f"{var}_bias "
    if err:
        out += f"{var}_error "
    if sens:
        out += f"{var}_sensitivity "
    return out[:-1]


def _cat_cnet_vars(vars_in, radar_meta, instruments):
    """Creates list of variable instances for output writing."""
    lin, log = 'linear', 'logarithmic'
    radar_source = instruments['radar']
    model_source = 'HYSPLIT'
    obs = []
    # dimensions and site location
    var = 'height'
    obs.append(CV(var, vars_in[var],
                  size=('height'),
                  fill_value=None,
                  long_name='Height above mean sea level',
                  units='m'))
    var = 'time'
    obs.append(CV(var, vars_in[var],
                  size=('time'),
                  fill_value=None,
                  long_name='Time UTC',
                  units='hours since ' + radar_meta['date'] + ' 00:00:00 +0:00'))
                  #comment='Fixed ' + str(config.TIME_RESOLUTION) + 's resolution.'))
    var = 'model_height'
    obs.append(CV(var, vars_in[var],
                  fill_value=None,
                  size=('model_height'),
                  long_name='Height of model variables above mean sea level',
                  units='m'))
    var = 'model_time'
    obs.append(CV(var, vars_in[var],
                  fill_value=None,
                  size=('model_time'),
                  long_name='Model time UTC',
                  units='hours since ' + radar_meta['date'] + ' 00:00:00 +0:00'))
    var = 'latitude'
    obs.append(CV(var, vars_in[var],
                  long_name='Latitude of site',
                  units='degrees_north'))
    var = 'longitude'
    obs.append(CV(var, vars_in[var],
                  long_name='Longitude of site',
                  units='degrees_east'))
    var = 'altitude'
    obs.append(CV(var, vars_in[var],
                  long_name='Altitude of site',
                  units='m',
                  comment=_comments(var)))
    # radar variables
    var = 'radar_frequency'
    obs.append(CV(var, vars_in[var],
                  long_name='Transmit frequency',
                  units='GHz'))
    var = 'Z'
    lname = 'Radar reflectivity factor'
    obs.append(CV(var, vars_in[var],
                  long_name=lname,
                  units='dBZ',
                  plot_range=(-40, 20),
                  plot_scale=lin,
                  comment=_comments(var),
                  source=radar_source,
                  ancillary_variables=_anc_names(var, True, True, True)))
    var = 'Z_bias'
    obs.append(CV(var, vars_in[var],
                  long_name=output.bias_name(lname),
                  units='dB',
                  comment=_comments('bias')))
    var = 'Z_error'
    obs.append(CV(var, vars_in[var],
                  long_name=output.err_name(lname),
                  units='dB',
                  comment=_comments(var)))
    var = 'Z_sensitivity'
    obs.append(CV(var, vars_in[var],
                  size=('height'),
                  long_name='Minimum detectable radar reflectivity',
                  units='dBZ',
                  comment=_comments(var)))
    var = 'v'
    obs.append(CV(var, vars_in[var],
                  long_name='Doppler velocity',
                  units='m s-1',
                  plot_range=(-4, 2),
                  plot_scale=lin,
                  comment=_comments(var),
                  source=radar_source))
    var = 'width'
    obs.append(CV(var, vars_in[var],
                  long_name='Spectral width',
                  units='m s-1',
                  plot_range=(0.03, 3),
                  plot_scale=log,
                  comment=_comments(var),
                  source=radar_source))
    var = 'ldr'
    obs.append(CV(var, vars_in[var],
                  long_name='Linear depolarisation ratio',
                  units='dB',
                  plot_range=(-30, 0),
                  plot_scale=lin,
                  comment=_comments(var),
                  source=radar_source))
    # lidar variables
    var = 'lidar_wavelength'
    obs.append(CV(var, vars_in[var],
                  long_name='Laser wavelength',
                  units='nm'))
    var = 'beta'
    lname = 'Attenuated backscatter coefficient'
    obs.append(CV(var, vars_in[var],
                  long_name=lname,
                  units='sr-1 m-1',
                  plot_range=(1e-7, 1e-4),
                  plot_scale=log,
                  source=instruments['lidar'],
                  ancillary_variables=_anc_names(var, bias=True, err=True)))
    var = 'beta_bias'
    obs.append(CV(var, vars_in[var],
                  long_name=output.bias_name(lname),
                  units='dB',
                  comment=_comments('bias')))
    var = 'beta_error'
    obs.append(CV(var, vars_in[var],
                  long_name=output.err_name(lname),
                  units='dB'))
    # mwr variables
    var = 'lwp'
    lname = 'Liquid water path'
    obs.append(CV(var, vars_in[var],
                  size=('time'),
                  long_name=lname,
                  units='g m-2',
                  plot_range=(-100, 1000),
                  plot_scale=lin,
                  source=instruments['mwr']))
    var = 'lwp_error'
    obs.append(CV(var, vars_in[var],
                  size=('time'),
                  long_name=output.err_name(lname),
                  units='g m-2'))
    # model variables
    var = 'temperature'
    obs.append(CV(var, vars_in[var],
                  size=('model_time', 'model_height'),
                  long_name='Temperature',
                  units='K',
                  plot_range=(200, 300),
                  plot_scale=lin,
                  source=model_source))
    var = 'pressure'
    obs.append(CV(var, vars_in[var],
                  size=('model_time', 'model_height'),
                  long_name='Pressure',
                  units='Pa',
                  plot_range=(0, 1.1e5),
                  plot_scale=log,
                  source=model_source))
    var = 'specific_humidity'
    obs.append(CV(var, vars_in[var],
                  size=('model_time', 'model_height'),
                  long_name='Model specific humidity',
                  plot_range=(0, 0.006),
                  plot_scale=lin,
                  source=model_source))
    var = 'uwind'
    obs.append(CV(var, vars_in[var],
                  size=('model_time', 'model_height'),
                  long_name='Zonal wind',
                  units='m s-1',
                  plot_range=(-50, 50),
                  plot_scale=lin,
                  source=model_source))
    var = 'vwind'
    obs.append(CV(var, vars_in[var],
                  size=('model_time', 'model_height'),
                  long_name='Meridional wind',
                  units='m s-1',
                  plot_range=(-50, 50),
                  plot_scale=lin,
                  source=model_source))
    # other
    var = 'Tw'
    obs.append(CV(var, vars_in[var],
                  fill_value=None,
                  long_name='Wet bulb temperature',
                  units='K',
                  comment=_comments(var)))
    var = 'insect_probability'
    obs.append(CV(var, vars_in[var],
                  fill_value=None,
                  long_name='Probability of insects'))
    var = 'radar_gas_atten'
    obs.append(CV(var, vars_in[var],
                  long_name='Two-way radar attenuation due to atmospheric gases',
                  units='dB',
                  plot_range=(0, 4),
                  plot_scale=lin,
                  comment=_comments(var)))
    var = 'radar_liquid_atten'
    obs.append(CV(var, vars_in[var],
                  long_name='Approximate two-way radar attenuation due to liquid water',
                  units='dB',
                  plot_range=(0, 4),
                  plot_scale=lin,
                  comment=_comments(var)))

    var = 'category_bits'
    obs.append(CV(var, vars_in[var],
                  data_type='i4',
                  fill_value=None,
                  long_name='Target classification bits',
                  valid_range=[0, 5],
                  comment=_comments(var),
                  definition=
                  ('\nBit 0: Small liquid droplets are present.\n'
                   'Bit 1: Falling hydrometeors are present; if Bit 2 is set then these are most\n'
                   '       likely ice particles, otherwise they are drizzle or rain drops.\n'
                   'Bit 2: Wet-bulb temperature is less than 0 degrees C, implying\n'
                   '       the phase of Bit-1 particles.\n'
                   'Bit 3: Melting ice particles are present.\n'
                   'Bit 4: Aerosol particles are present and visible to the lidar.\n'
                   'Bit 5: Insects are present and visible to the radar.')))
    var = 'quality_bits'
    obs.append(CV(var, vars_in[var],
                  data_type='i4',
                  fill_value=None,
                  long_name='Data quality bits',
                  valid_range=[0, 5],
                  comment=_comments(var),
                  definition=
                  ('\nBit 0: An echo is detected by the radar.\n'
                   'Bit 1: An echo is detected by the lidar.\n'
                   'Bit 2: The apparent echo detected by the radar is ground clutter\n'
                   '       or some other non-atmospheric artifact.\n'
                   'Bit 3: The lidar echo is due to clear-air molecular scattering.\n'
                   'Bit 4: Liquid water cloud, rainfall or melting ice below this pixel\n'
                   '       will have caused radar and lidar attenuation; if bit 5 is set then\n'
                   '       a correction for the radar attenuation has been performed;\n'
                   '       otherwise do not trust the absolute values of reflectivity factor.\n'
                   '       No correction is performed for lidar attenuation.\n'
                   'Bit 5: Radar reflectivity has been corrected for liquid-water attenuation\n'
                   '       using the microwave radiometer measurements of liquid water path\n'
                   '       and the lidar estimation of the location of liquid water cloud;\n'
                   '       be aware that errors in reflectivity may result.')))
    return obs


def _comments(field):
    """Returns the comment text for a Cloudnet variable."""
    com = {
        'category_bits':
        ('This variable contains information on the nature of the targets\n'
         'at each pixel, thereby facilitating the application of algorithms that work\n'
         'with only one type of target. The information is in the form of an array of\n'
         'bits, each of which states either whether a certain type of particle is present\n'
         '(e.g. aerosols), or the whether some of the target particles have a particular\n'
         'property. The definitions of each bit are given in the definition attribute.\n'
         'Bit 0 is the least significant.'),

        'quality_bits':
        ('This variable contains information on the quality of the\n'
         'data at each pixel. The information is in the form of an array\n'
         'of bits, and the definitions of each bit are given in the definition\n'
         'attribute. Bit 0 is the least significant'),

        'radar_liquid_atten':
        ('This variable was calculated from the liquid water path\n'
         'measured by microwave radiometer using lidar and radar returns to perform\n'
         'an approximate partioning of the liquid water content with height. Bit 5 of\n'
         'the quality_bits variable indicates where a correction for liquid water\n'
         'attenuation has been performed.'),

        'radar_gas_atten':
        ('This variable was calculated from the model temperature,\n'
         'pressure and humidity, but forcing pixels containing liquid cloud to saturation\n'
         'with respect to liquid water. It was calculated using the millimeter-wave propagation\n'
         'model of Liebe (1985, Radio Sci. 20(5), 1069-1089). It has been used to correct Z.'),

        'Tw':
        ('Calculated from model T, P and relative humidity, which were first\n'
         'interpolated into measurement grid.'),

        'Z_sensitivity':
        ('This variable is an estimate of the radar sensitivity,\n'
         'i.e. the minimum detectable radar reflectivity, as a function\n'
         'of height. It includes the effect of ground clutter and gas attenuation\n'
         'but not liquid attenuation.'),

        'Z_error':
        ('This variable is an estimate of the one-standard-deviation\n'
         'random error in radar reflectivity factor. It originates\n'
         'from the following independent sources of error:\n'
         '1) Precision in reflectivity estimate due to finite signal to noise\n'
         '   and finite number of pulses\n'
         '2) 10% uncertainty in gaseous attenuation correction (mainly due to\n'
         '   error in model humidity field)\n'
         '3) Error in liquid water path (given by the variable lwp_error) and\n'
         '   its partitioning with height).'),

        'altitude':
        ('Defined as the altitude of radar or lidar - the one that is lower.'),

        'Z':
        ('This variable has been corrected for attenuation by gaseous\n'
         'attenuation (using the thermodynamic variables from a forecast\n'
         'model; see the radar_gas_atten variable) and liquid attenuation\n'
         '(using liquid water path from a microwave radiometer; see the\n'
         'radar_liquid_atten variable) but rain and melting-layer attenuation\n'
         'has not been corrected. Calibration convention: in the absence of\n'
         'attenuation, a cloud at 273 K containing one million 100-micron droplets\n'
         'per cubic metre will have a reflectivity of 0 dBZ at all frequencies.'),

        'bias':
        ('This variable is an estimate of the one-standard-deviation calibration error.'),

        'ldr':
        ('This parameter is the ratio of cross-polar to co-polar reflectivity.'),

        'width':
        ('This parameter is the standard deviation of the reflectivity-weighted\n'
         'velocities in the radar pulse volume.'),

        'v':
        ('This parameter is the radial component of the velocity, with positive\n'
         'velocities are away from the radar.'),        
        
    }
    return com[field]
