""" Functions for rebinning input data.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('../../cloudnetpy'))
import math
import numpy as np
import numpy.ma as ma
import scipy.constants
from scipy.interpolate import interp1d
from cloudnetpy import config
from cloudnetpy import ncf
from cloudnetpy import utils
from cloudnetpy import atmos
from cloudnetpy import classify
from cloudnetpy import output
from cloudnetpy import plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from cloudnetpy.output import CloudnetVariable
import sys
import netCDF4

#class GlobalMeta():
#    def __init__(self, radar_file, lidar_file):
#        pass
        #self.site_altitude =
        #self.date =

class CloudnetDataSource():
    """Base class for all NetCDF files."""
    def __init__(self, netcdf_file):
        self.dataset = netCDF4.Dataset(netcdf_file)
        self.variables = self.dataset.variables
        self._copy_attributes(('location', 'source'))

    def netcdf_to_cloudnet(self, fields):
        """Transforms NetCDF variables (data + attributes) to CloudnetVariables."""
        self.data = {}
        for name in fields:
            self.data[name] = output.CloudnetVariable(self.variables[name], name)

    def _get_height(self):
        """Returns height above mean sea level."""
        range_instru = ncf.km2m(self.variables['range'])
        alt_instru = ncf.km2m(self.variables['altitude'])
        return np.array(range_instru + alt_instru)
        
    def _copy_attributes(self, names):
        for name in names:
            if hasattr(self.dataset, name):
                value = getattr(self.dataset, name)
                setattr(self, name, value)


class Radar(CloudnetDataSource):
    """Class for radar data."""
    def __init__(self, radar_file):
        super().__init__(radar_file)
        self.frequency = ncf.radar_freq(self.variables)
        self.wl_band = ncf.wl_band(self.frequency)
        self.folding_velocity = ncf.folding_velo(self.variables,
                                                 self.frequency)
        self.height = self._get_height()
        self.altitude = self.variables['altitude'][:]


class Lidar(CloudnetDataSource):
    """Class for lidar data."""
    def __init__(self, lidar_file):
        super().__init__(lidar_file)
        self.height = self._get_height()


class Mwr(CloudnetDataSource):
    """Class for microwaver radiometer data."""
    def __init__(self, mwr_file):
        super().__init__(mwr_file)
        self.lwp_name = ncf.findkey(self.variables, ('LWP_data', 'lwp'))


class Model(CloudnetDataSource):
    """Class for model data."""
    fields_dense = ('temperature', 'pressure', 'rh',
                    'gas_atten', 'specific_gas_atten',
                    'specific_saturated_gas_atten',
                    'specific_liquid_atten')
    fields_all = fields_dense + ('q', 'uwind', 'vwind')
    
    def __init__(self, model_file, alt_site):
        super().__init__(model_file)
        self.time = self.variables['time'][:]
        self.model_heights = self._get_model_heights(alt_site)
        self.mean_height = self._get_mean_height()

    def _get_model_heights(self, alt_site):
        return ncf.km2m(self.variables['height']) + alt_site
        
    def _get_mean_height(self):
        return np.mean(np.array(self.model_heights), axis=0)
        
    def interpolate_to_common_height(self, wl_band, field_names):
        """Interpolates model variables to common height grid."""

        def _interpolate_variable(data):
            datai = np.zeros((len(self.time), len(self.mean_height)))
            for ind, (alt, prof) in enumerate(zip(self.model_heights, data)):
                f = interp1d(alt, prof, fill_value='extrapolate')
                datai[ind, :] = f(self.mean_height)
            return datai
        
        self.data_sparse = {}
        for key in field_names:
            data = np.array(self.variables[key][:])
            if 'atten' in key:
                data = data[wl_band, :, :]
            self.data_sparse[key] = _interpolate_variable(data)

    def interpolate_to_cloudnet_grid(self, field_names, *newgrid):
        self.data_dense = {}
        for key in field_names:
            self.data_dense[key] = utils.interpolate_2d(self.time,
                                                        self.mean_height,
                                                        *newgrid,
                                                        self.data_sparse[key])

    def calc_wet_bulb(self):
        self.Tw = atmos.wet_bulb(self.data_dense['temperature'],
                                 self.data_dense['pressure'],
                                 self.data_dense['rh'])
        

def generate_categorize(input_files, output_file, zlib=True):

    #meta = GlobalMeta(*input_files[0:2])
    
    radar = Radar(input_files[0])
    lidar = Lidar(input_files[1])
    mwr = Mwr(input_files[2])
    model = Model(input_files[3], radar.altitude)

    # new grid
    time = utils.time_grid()
    height = radar.height
    
    # measurement data
    radar.netcdf_to_cloudnet(('Zh', 'v', 'ldr', 'width'))
    lidar.netcdf_to_cloudnet(('beta', 'beta_raw'))
    mwr.netcdf_to_cloudnet((mwr.lwp_name,))

    # model data
    model.netcdf_to_cloudnet(model.fields_all)
    model.interpolate_to_common_height(radar.wl_band, model.fields_all)
    model.interpolate_to_cloudnet_grid(model.fields_dense, time, height)
    model.calc_wet_bulb()



    
    #input_types = ncf.fetch_input_types(input_files)
    #time = utils.time_grid()
    #height = _height_above_sea(rad_vars)
    #radar_meta = ncf.fetch_radar_meta(input_files[0])
    #alt_site = ncf.site_altitude(rad_vars, lid_vars, mwr_vars)
    #radar = fetch_data(rad_vars, ('Zh', 'v', 'ldr', 'width'), time, vfold=radar_meta['vfold'])
    #lidar = fetch_data(lid_vars, ('beta', 'beta_raw'), time, height_new=height)
    #lwp = fetch_mwr(mwr_vars, config.LWP_ERROR, time)
    #model = fetch_model(mod_vars, alt_site, radar_meta['freq'], time, height)

    bits = classify.fetch_cat_bits(radar, lidar['beta'], model['Tw'], time, height, input_types['model'])
    gas_atten = atmos.gas_atten(model['interp'], bits['cat'], height)
    liq_atten = atmos.liquid_atten(lwp, model['interp'], bits, height)
    qual_bits = classify.fetch_qual_bits(radar['Zh'], lidar['beta'], bits['clutter'], liq_atten)
    Z_corrected = _correct_atten(radar['Zh'], gas_atten, liq_atten['value'])
    Z_err = _fetch_Z_errors(radar, rad_vars, gas_atten, liq_atten,
                            bits['clutter'], radar_meta['freq'],
                            time, config.GAS_ATTEN_PREC)

    
    cat_vars = {
        'height': height,
        'time': time,
        'latitude': float(rad_vars['latitude'][:]),
        'longitude': float(rad_vars['longitude'][:]),
        'altitude': float(alt_site),
        'radar_frequency': radar_meta['freq'],
        'lidar_wavelength': float(lid_vars['wavelength'][:]),
        'beta': lidar['beta'],
        'beta_raw': lidar['beta_raw'],
        'beta_error': config.BETA_ERROR[0],
        'beta_bias': config.BETA_ERROR[1],
        'Z': Z_corrected,
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
    obs = output.create_objects_for_output(cat_vars)
    output.save_cat(output_file, time, height, model['time'],
                    model['height'], obs, radar_meta, zlib)


def _correct_atten(Z, gas_atten, liq_atten):
    """Corrects radar echo for attenuation.

    Args:
        Z (MaskedArray): 2-D array of radar echo.
        gas_atten (ndarray): 2-D array of attenuation due to atmospheric gases.
        liq_atten (MaskedArray): 2-D array of attenuation due to atmospheric liquid.

    Returns:
        MaskedArray: Copy of *Z*, corrected by liquid attenuation
        (where applicable) and gas attenuation (everywhere).

    """
    Z_corr = ma.copy(Z) + gas_atten
    ind = ~liq_atten.mask
    Z_corr[ind] = Z_corr[ind] + liq_atten[ind]
    return Z_corr


def fetch_data(vars_in, fields, time_new, height_new=None, vfold=None):
    """Reads and rebins radar / lidar 2-D fields in time.

    Args:
        vars_in (dict): Measured variables.
        fields (tuple): Tuple of strings containing 2-D fields to be
            rebinned, e.g. ('Zh', 'v', 'width') or ('beta', 'beta_raw').
        time_new (ndarray): 1-D array, the target time vector.
        height_new (ndarray, optional): 1-D array, the target height vector.
        vfold (float, optional): Radar folding velocity = Pi/NyquistVelocity (m/s).

    Returns:
        dict: Rebinned fields.

    Raises:
        KeyError: Missing field.

    Notes:
        Radar echo, 'Zh', is averaged in linear space.
        Doppler velocity, 'v', is averaged in polar coordinates.

    """
    out = {}
    time_instru = vars_in['time'][:]
    for field in fields:
        if field not in vars_in:
            raise KeyError(f"No variable '{field}' in the radar file.")
        data = vars_in[field][:]
        if field == 'Zh':  # average in linear scale
            data_lin = utils.db2lin(data)
            data_mean = utils.rebin_2d(time_instru, data_lin, time_new)
            out[field] = utils.lin2db(data_mean)
        elif field == 'v':  # average in polar coordinates
            data = data * vfold
            vx, vy = np.cos(data), np.sin(data)
            vx_mean = utils.rebin_2d(time_instru, vx, time_new)
            vy_mean = utils.rebin_2d(time_instru, vy, time_new)
            out[field] = np.arctan2(vy_mean, vx_mean) / vfold
        elif field in ('beta', 'beta_raw'):  # average in time and altitude
            height_lidar = _height_above_sea(vars_in)
            data = utils.rebin_2d(time_instru, vars_in[field][:], time_new)
            out[field] = utils.rebin_2d(height_lidar, data.T, height_new).T
        else:  # average in time, no conversions
            out[field] = utils.rebin_2d(time_instru, data, time_new)
    return out


def fetch_mwr(mwr_vars, lwp_errors, time):
    """Returns interpolated liquid water path and its error.

    Args:
        mwr_vars (dict): Radiometer variables.
        lwp_errors (tuple): 2-element tuple containing
            (fractional_error, linear_error)
        time (ndarray): 1-D array, the target time vector.

    Returns:
        Dict containing

        - **value** (*ndarray*): Interpolated LWP.
        - **err** (*ndarray*): Error of LWP.

    Notes:
        Needs to decide how to handle totally
        missing (or sparse) mwr data.

    """
    def _interpolate_lwp(time_lwp, data, time_new):
        """Linear interpolation of LWP data."""
        try:
            f = interp1d(time_lwp, data)
            data_interp = f(time_new)
        except:
            data_interp = np.full_like(time_new, fill_value=np.nan)
        return data_interp

    data, time_lwp, error = _read_lwp(mwr_vars, *lwp_errors)
    data_interp = _interpolate_lwp(time_lwp, data, time)
    error_interp = _interpolate_lwp(time_lwp, error, time)
    return {'value': data_interp, 'err': error_interp}


def _read_lwp(mwr_vars, frac_err, lin_err):
    """Reads LWP, estimates its error, and converts time if needed.

    Args:
        mwr_vars (dict): Radiometer variables.
        frac_error (float): Fractional error (scalar).
        lin_error (float): Linear error (scalar).

    Returns:
        tuple: 3-element tuple containing liquid water path
            variables (data, time, error).

    Note:
        hatpro time can be 'hours since' 00h of measurement date
        or 'seconds since' some epoch.

    """
    data_field = ncf.findkey(mwr_vars, ('LWP_data', 'lwp'))
    data = mwr_vars[data_field][:]
    time = mwr_vars['time'][:]
    if max(time) > 24:
        time = utils.seconds2hour(time)
    error = utils.l2norm(frac_err*data, lin_err)
    return data, time, error


def _fetch_Z_errors(radar, rad_vars, gas_atten, liq_atten,
                    is_clutter, freq, time, gas_atten_prec):
    """Calculates sensitivity and error of radar echo.

    Args:
        radar (dict): Interpolated radar variables.
        rad_vars (dict): Original radar variables.
        gas_atten (ndarray): 2-D gas attenuation.
        liq_atten (dict): Liquid attenuation error and boolean
            arrays denoting where liquid attenuation was not
            corrected {'err', 'is_not_corr'}.
        is_clutter (ndarray): Boolean array denoting pixels
            contaminated by clutter.
        freq (float): Radar frequency (GHz).
        time (ndarray): 1-D time vector.
        gas_atten_prec (float): Precision of gas attenuation
            between 0 and 1, e.g., 0.1.

    Returns:
        dict: Error-related variables {'Z_sensitivity', 'Z_error'}
            which are 1-D and 2-D MaskedArrays, respectively.

    """
    Z = radar['Zh']
    radar_range = ncf.km2m(rad_vars['range'])
    log_range = utils.lin2db(radar_range, scale=20)
    Z_power = Z - log_range
    Z_power_min = np.percentile(Z_power.compressed(), 0.1)
    Z_sensitivity = Z_power_min + log_range + np.mean(gas_atten, axis=0)
    Zc = ma.median(ma.array(Z, mask=~is_clutter), axis=0)
    Z_sensitivity[~Zc.mask] = Zc[~Zc.mask]
    dwell_time = utils.mdiff(time)*3600  # seconds
    independent_pulses = (dwell_time*freq*1e9*4*np.sqrt(math.pi)
                          * radar['width']/3e8)
    Z_precision = 4.343*(1/np.sqrt(independent_pulses)
                         + utils.db2lin(Z_power_min-Z_power)/3)
    Z_error = utils.l2norm(gas_atten*gas_atten_prec, liq_atten['err'],
                           Z_precision)
    Z_error[liq_atten['is_not_corr']] = ma.masked
    return {'sensitivity': Z_sensitivity, 'error': Z_error}
