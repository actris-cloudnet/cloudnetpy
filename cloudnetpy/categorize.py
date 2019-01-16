""" Functions for rebinning input data.
"""
import os
import sys
sys.path.insert(0, os.path.abspath('../../cloudnetpy'))
import math
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
import netCDF4
from cloudnetpy import config
from cloudnetpy import utils
from cloudnetpy import atmos
from cloudnetpy import classify
from cloudnetpy import output
from cloudnetpy import plotting
from cloudnetpy.cloudnetarray import CloudnetArray

class RawDataSource():
    """Base class for all Cloudnet measurements and model data.

    Attributes:
        filename: Filename of the input file.
        dataset: A netcdf4 Dataset instance.
        variables: Variables of the Dataset instance.
        source: Global attribute 'source' from input file.
        time: The time vector.
    """
    def __init__(self, filename):
        self.filename = filename
        self.dataset = netCDF4.Dataset(self.filename)
        self.variables = self.dataset.variables
        self.source = self._get_global_attribute('source')
        self.time = self._getvar(('time',))
        self.data = {}

    def netcdf_to_cloudnet(self, fields):
        """Transforms NetCDF variables (data + attributes) into CloudnetArrays."""
        for name in fields:
            self.data[name] = CloudnetArray(self.variables[name], name)

    def _get_height(self):
        """Returns height above mean sea level."""
        range_instru = utils.km2m(self.variables['range'])
        alt_instru = utils.km2m(self.variables['altitude'])
        return np.array(range_instru + alt_instru)

    def _get_global_attribute(self, attr_name):
        """Returns attribute from the source file."""
        if hasattr(self.dataset, attr_name):
            return getattr(self.dataset, attr_name)
        return ''

    def _getvar(self, possible_names):
        """Returns variable from the source file."""
        matching_key = utils.findkey(self.variables, possible_names)
        if not matching_key:
            raise KeyError('Missing variable.')
        return self.variables[matching_key][:]


class Radar(RawDataSource):
    """Class for radar data.

    Child of RawDataSource class. Contains methods for radar data processing.

    Attributes:
        frequency (float): Radar frequency (GHz).
        wl_band (int): Int corresponding to frequency 0 = 35.5 GHz, 1 = 94 GHz.
        folding_velocity (float): Radar's folding velocity (m/s).
        height (ndarray): Measurement height grid above mean sea level (m).
        altitude (float): Altitude of the radar above mean sea level (m).

    """
    def __init__(self, radar_file, fields):
        super().__init__(radar_file)
        self.frequency = self._getvar(('radar_frequency', 'frequency'))
        self.wl_band = self._get_wl_band()
        self.folding_velocity = self._get_folding_velo()
        self.height = self._get_height()
        self.altitude = self._getvar(('altitude',))
        self.netcdf_to_cloudnet(fields)

    def rebin_data(self, time):
        """Rebins radar data using mean."""
        for variable in self.data:
            if variable in ('Zh',):
                self.data[variable].db2lin()
                self.data[variable].rebin_data(self.time, time)
                self.data[variable].lin2db()
            elif variable in ('v',):
                self.data[variable].rebin_in_polar(self.time, time,
                                                   self.folding_velocity)
            else:
                self.data[variable].rebin_data(self.time, time)

    def _get_wl_band(self):
        return 0 if (30 < self.frequency < 40) else 1

    def _get_folding_velo(self):
        if 'NyquistVelocity' in self.variables:
            nyquist = self.variables['NyquistVelocity'][:]
        elif 'prf' in self.variables:
            nyquist = self.variables['prf'][:] * scipy.constants.c / (4 * self.frequency)
        return math.pi / nyquist


class Lidar(RawDataSource):
    """Class for lidar data.

    Child of RawDataSource class. Contains
    methods for lidar data processing.

    Attributes:
        height (ndarray): Altitude grid above mean sea level (m).

    """
    def __init__(self, lidar_file, fields):
        super().__init__(lidar_file)
        self.height = self._get_height()
        self.netcdf_to_cloudnet(fields)

    def rebin_data(self, time, height):
        """Rebins lidar data in time and height."""
        for variable in self.data:
            self.data[variable].rebin_data(self.time, time, self.height, height)


class Mwr(RawDataSource):
    """Class for microwaver radiometer data.

    Child of RawDataSource class. Contains
    methods for microwave radiometer processing.

    Attributes:
        lwp_name (str): Name of the data field in lwp-file, e.g.
            'LWP_data' or 'lwp'.
        error (ndarray): Error estimate of lwp.
        time (ndarray): Time vector of lwp.
        data (dict): ClounetVariable objects.

    """
    def __init__(self, mwr_file):
        super().__init__(mwr_file)
        self.lwp_name = utils.findkey(self.variables, ('LWP_data', 'lwp'))
        self.netcdf_to_cloudnet((self.lwp_name,))
        self.error = self._calc_lwp_error(*config.LWP_ERROR)
        self._fix_time()

    def interpolate_to_cloudnet_grid(self, time):
        """Interpolates liquid water path to Cloudnets dense time grid."""
        f = interp1d(self.time, self.data[self.lwp_name][:])
        self.data[self.lwp_name] = CloudnetArray(f(time), 'lwp')
        self.time = time
        
    def _calc_lwp_error(self, fractional_error, linear_error):
        lwp = self.data[self.lwp_name].data
        error = utils.l2norm(lwp*fractional_error, linear_error)
        return CloudnetArray(error, 'lwp_error')

    def _fix_time(self):
        if max(self.time) > 24:
            self.time = utils.seconds2hour(self.time)


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
        self.data_sparse = None
        self.data_dense = None
        self.Tw = None
        self.time = self._getvar(('time',))
        
    def _get_model_heights(self, alt_site):
        return utils.km2m(self.variables['height']) + alt_site

    def _get_mean_height(self):
        return np.mean(np.array(self.model_heights), axis=0)

    def interpolate_to_common_height(self, wl_band, field_names):
        """Interpolates model variables to common height grid."""

        def _interpolate_variable(data, key):
            datai = np.zeros((len(self.time), len(self.mean_height)))
            for ind, (alt, prof) in enumerate(zip(self.model_heights, data)):
                f = interp1d(alt, prof, fill_value='extrapolate')
                datai[ind, :] = f(self.mean_height)
            return CloudnetArray(datai, key)

        self.data_sparse = {}
        for key in field_names:
            data = np.array(self.variables[key][:])
            if 'atten' in key:
                data = data[wl_band, :, :]
            self.data_sparse[key] = _interpolate_variable(data, key)

        for name in self.data:
            print(name, self.data[name].units)

            
    def interpolate_to_cloudnet_grid(self, field_names, *newgrid):
        """Interpolates model variables to Cloudnets dense time / height grid."""
        self.data_dense = {}
        for key in field_names:
            self.data_dense[key] = utils.interpolate_2d(self.time,
                                                        self.mean_height,
                                                        *newgrid,
                                                        self.data_sparse[key][:])

    def calc_wet_bulb(self):
        """Calculates wet-bulb temperature in dense grid."""
        Tw = atmos.wet_bulb(self.data_dense['temperature'],
                            self.data_dense['pressure'],
                            self.data_dense['rh'])
        self.data['Tw'] = CloudnetArray(Tw, 'Tw', units='K')

def _interpolate_to_cloudnet_grid(radar, lidar, mwr, model, grid):
    """ Interpolate variables to Cloudnet's dense grid."""
    model.interpolate_to_common_height(radar.wl_band, model.fields_all)
    model.interpolate_to_cloudnet_grid(model.fields_dense, *grid)
    mwr.interpolate_to_cloudnet_grid(grid[0])
    radar.rebin_data(grid[0])
    lidar.rebin_data(*grid)
    model.calc_wet_bulb()

        
def generate_categorize(input_files, output_file, zlib=True):
    """ High level API to generate Cloudnet categorize file.

    """
    radar = Radar(input_files[0], ('Zh', 'v', 'ldr', 'width'))
    lidar = Lidar(input_files[1], ('beta',))
    mwr = Mwr(input_files[2])
    model = Model(input_files[3], radar.altitude)
    grid = (utils.time_grid(), radar.height)
    _interpolate_to_cloudnet_grid(radar, lidar, mwr, model, grid)

    cbits, liquid_bases, is_rain, is_clutter = classify.classify_measurements(radar.data,
                                                                              lidar.data['beta'][:],
                                                                              model.data['Tw'][:],
                                                                              grid, 'gdas')

    output_data = {**radar.data, **lidar.data, **model.data_sparse,
                   **mwr.data, **cbits}
    output.update_attributes(output_data)

    _save_cat(output_file, grid, (model.time, model.mean_height), output_data)

    



def _save_cat(file_name, grid, model_grid, obs):
    """Creates a categorize netCDF4 file and saves all data into it."""
    rootgrp = netCDF4.Dataset(file_name, 'w', format='NETCDF4_CLASSIC')
    # create dimensions
    time = rootgrp.createDimension('time', len(grid[0]))
    height = rootgrp.createDimension('height', len(grid[1]))
    model_time = rootgrp.createDimension('model_time', len(model_grid[0]))
    model_height = rootgrp.createDimension('model_height', len(model_grid[1]))
    # root group variables
    output.write_vars2nc(rootgrp, obs, zlib=True)
    # global attributes:
    rootgrp.Conventions = 'CF-1.7'
    #rootgrp.title = 'Categorize file from ' + radar_meta['location']
    #rootgrp.institution = 'Data processed at the ' + config.INSTITUTE
    #dvec = radar_meta['date']
    #rootgrp.year = int(dvec[:4])
    #rootgrp.month = int(dvec[5:7])
    #rootgrp.day = int(dvec[8:])
    #rootgrp.software_version = version
    #rootgrp.git_version = ncf.git_version()
    #rootgrp.file_uuid = str(uuid.uuid4().hex)
    #rootgrp.references = 'https://doi.org/10.1175/BAMS-88-6-883'
    #rootgrp.history = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} - categorize file created"
    rootgrp.close()

    
    
    """
    
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

    """

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
    radar_range = utils.km2m(rad_vars['range'])
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


