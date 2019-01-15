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


class RawDataSource():
    """Base class for all Cloudnet measurements and model data.

    Attributes:
        filename: A netcdf4 file. 
        dataset: A netcdf4 Dataset instance.
        variables: Variables of the Dataset instance.
        source: Global attribute 'source' from input file.
        time: The time vector.
    """
    def __init__(self, filename):
        self.filename = filename
        self.dataset = netCDF4.Dataset(self.filename)
        self.variables = self.dataset.variables
        self.source = self._copy_attribute('source')
        self.time = self.variables['time'][:]

    def netcdf_to_cloudnet(self, fields):        
        """Transforms NetCDF variables (data + attributes) into CloudnetVariables."""
        self.data = {}
        for name in fields:
            self.data[name] = output.CloudnetVariable(self.variables[name], name)

    def _get_height(self):
        """Returns height above mean sea level."""
        range_instru = ncf.km2m(self.variables['range'])
        alt_instru = ncf.km2m(self.variables['altitude'])
        return np.array(range_instru + alt_instru)

    def _copy_attribute(self, attr_name):
        """Returns global attribute from the source file if the attribute exists."""
        if hasattr(self.dataset, attr_name):
            return getattr(self.dataset, attr_name)

        
class Radar(RawDataSource):
    """Class for radar data.

    Child of CloudnetDataSource class. Contains
    methods for radar data processing.

    Attributes:
        frequency (float): Radar frequency (GHz).
        wl_band (int): Int corresponding to frequency 0 = 35.5 GHz, 
            1 = 94 GHz.
        folding_velocity (float): Radar's folding velocity (m/s).
        height (ndarray): Measurement height grid above mean sea level (m).
        altitude (float): Altitude of the radar above mean sea level (m).

    """
    def __init__(self, radar_file):
        super().__init__(radar_file)
        self.frequency = ncf.radar_freq(self.variables)
        self.wl_band = ncf.wl_band(self.frequency)
        self.folding_velocity = ncf.folding_velo(self.variables,
                                                 self.frequency)
        self.height = self._get_height()
        self.altitude = self.variables['altitude'][:]

    def rebin_data(self, time):
        for variable in self.data:
            if variable in ('Zh',):
                self.data[variable].db2lin()
                self.data[variable].rebin_data(self.time, time)
                self.data[variable].lin2db()            
            elif variable in ('v',):
                self.data[variable].rebin_in_polar(self.time, time, self.folding_velocity)
            else:
                self.data[variable].rebin_data(self.time, time)

        
class Lidar(RawDataSource):
    """Class for lidar data."""
    def __init__(self, lidar_file):
        super().__init__(lidar_file)
        self.height = self._get_height()

    def rebin_data(self, time, height):
        for variable in self.data:
            self.data[variable].rebin_data(self.time, time, self.height, height)


class Mwr(RawDataSource):
    """Class for microwaver radiometer data."""
    def __init__(self, mwr_file):
        super().__init__(mwr_file)
        self.lwp_name = ncf.findkey(self.variables, ('LWP_data', 'lwp'))
        self.netcdf_to_cloudnet((self.lwp_name,))
        self.error = self._calc_lwp_error(*config.LWP_ERROR)
        self.time = self._get_time()
        
    def _calc_lwp_error(self, fractional_error, linear_error):
        lwp = self.data[self.lwp_name]._data
        error = utils.l2norm(lwp*fractional_error, linear_error)
        return CloudnetVariable(error, 'lwp_error')
        
    def _get_time(self):
        time = self.variables['time'][:]
        if max(time) > 24:
            time = utils.seconds2hour(time)
        return time
        
    def interpolate_to_cloudnet_grid(self, time):
        f = interp1d(self.time, self.data[self.lwp_name]._data)
        self.data = f(time)
        self.time = time
        
        
class Model(RawDataSource):
    """Class for model data."""
    fields_dense = ('temperature', 'pressure', 'rh',
                    'gas_atten', 'specific_gas_atten',
                    'specific_saturated_gas_atten',
                    'specific_liquid_atten')
    fields_all = fields_dense + ('q', 'uwind', 'vwind')
    
    def __init__(self, model_file, alt_site):
        super().__init__(model_file)
        self.model_heights = self._get_model_heights(alt_site)
        self.mean_height = self._get_mean_height()
        self.netcdf_to_cloudnet(self.fields_all)

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
        """Interpolates model variables to Cloudnets dense time / height grid."""
        self.data_dense = {}
        for key in field_names:
            self.data_dense[key] = utils.interpolate_2d(self.time,
                                                        self.mean_height,
                                                        *newgrid,
                                                        self.data_sparse[key])

    def calc_wet_bulb(self):
        """Calculates wet-bulb temperature in dense grid."""
        self.Tw = atmos.wet_bulb(self.data_dense['temperature'],
                                 self.data_dense['pressure'],
                                 self.data_dense['rh'])


def generate_categorize(input_files, output_file, zlib=True):

    # Construct instances.
    radar = Radar(input_files[0])
    lidar = Lidar(input_files[1])
    mwr = Mwr(input_files[2])
    model = Model(input_files[3], radar.altitude)
    
    # new grid
    time = utils.time_grid()
    height = radar.height
    
    radar.netcdf_to_cloudnet(('Zh', 'v', 'ldr', 'width'))
    lidar.netcdf_to_cloudnet(('beta',))
    model.interpolate_to_common_height(radar.wl_band, model.fields_all)
    model.interpolate_to_cloudnet_grid(model.fields_dense, time, height)    
    model.calc_wet_bulb()
    mwr.interpolate_to_cloudnet_grid(time)
    radar.rebin_data(time)
    lidar.rebin_data(time, height)

    sys.exit(0)
    
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
