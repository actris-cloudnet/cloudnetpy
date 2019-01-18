""" Functions for rebinning input data.
"""
#import os
#import sys
#sys.path.insert(0, os.path.abspath('../../cloudnetpy'))
import math
import numpy as np
import numpy.ma as ma
import scipy.constants
from scipy.interpolate import interp1d
import netCDF4
from cloudnetpy import config
from cloudnetpy import utils
from cloudnetpy import atmos
from cloudnetpy import classify
from cloudnetpy import output
from cloudnetpy.cloudnetarray import CloudnetArray
import cloudnetpy.version


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
        self.time = self._getvar('time')
        self.data = {}

    def _fix_time(self):
        if max(self.time) > 24:
            self.time = utils.seconds2hour(self.time)

    def _get_altitude(self):
        """Returns altitude of the instrument (m)."""
        return utils.km2m(self.variables['altitude'])

    def _get_height(self):
        """Returns height array above mean sea level (m)."""
        range_instrument = utils.km2m(self.variables['range'])
        alt_instrument = self._get_altitude()
        return np.array(range_instrument + alt_instrument)

    def _get_global_attribute(self, attr_name):
        """Returns attribute from the source file."""
        if hasattr(self.dataset, attr_name):
            return getattr(self.dataset, attr_name)
        return ''

    def _getvar(self, *args):
        """Returns data (without attributes) from the source file."""
        for arg in args:
            if arg in self.variables:
                return self.variables[arg][:]
        raise KeyError('Missing variable')

    def _append_data(self, data, key, name=None, units=None):
        self.data[key] = CloudnetArray(data, name or key, units)

    def netcdf_to_cloudnet(self, fields):
        """Transforms NetCDF variables into CloudnetArrays."""
        for key in fields:
            self._append_data(self.variables[key], key)


class Radar(RawDataSource):
    """Class for radar data.

    Attributes:
        frequency (float): Radar frequency (GHz).
        wl_band (int): Int corresponding to frequency 0 = 35.5 GHz, 1 = 94 GHz.
        folding_velocity (float): Radar's folding velocity (m/s).
        height (ndarray): Measurement height grid above mean sea level (m).
        altitude (float): Altitude of the radar above mean sea level (m).

    """
    def __init__(self, radar_file, fields):
        super().__init__(radar_file)
        self.radar_frequency = self._getvar('radar_frequency', 'frequency')
        self.wl_band = self._get_wl_band()
        self.folding_velocity = self._get_folding_velocity()
        self.altitude = self._get_altitude()
        self.height = self._get_height()
        self.netcdf_to_cloudnet(fields)
        self.location = self._get_global_attribute('location')

    def _get_wl_band(self):
        return 0 if (30 < self.radar_frequency < 40) else 1

    def _get_folding_velocity(self):
        if 'NyquistVelocity' in self.variables:
            nyquist = self._getvar('NyquistVelocity')
        elif 'prf' in self.variables:
            nyquist = (self._getvar('prf') * scipy.constants.c
                       / (4 * self.radar_frequency))
        else:
            raise KeyError('Unable to determine folding velocity')
        return math.pi / nyquist

    def rebin_data(self, time_new):
        """Rebins radar data in time using mean."""
        for key in self.data:
            if key in ('Zh',):
                self.data[key].db2lin()
                self.data[key].rebin_data(self.time, time_new)
                self.data[key].lin2db()
            elif key in ('v',):
                self.data[key].rebin_in_polar(self.time, time_new,
                                              self.folding_velocity)
            else:
                self.data[key].rebin_data(self.time, time_new)
        self.time = time_new

    def correct_atten(self, gas_atten, liq_atten):
        """Corrects radar echo for attenuation."""
        z_corrected = self.data['Zh'][:]
        z_corrected += gas_atten
        ind = ~liq_atten.mask
        z_corrected[ind] += liq_atten[ind]
        self._append_data(z_corrected, 'Z')

    def calc_errors(self, gas_atten, liq_atten_err, is_clutter,
                    is_uncorrected_liquid, time):
        """Calculates error and sensitivity of radar echo."""
        z = self.data['Zh'][:]
        radar_range = utils.km2m(self.variables['range'])
        log_range = utils.lin2db(radar_range, scale=20)
        z_power = z - log_range
        z_power_min = np.percentile(z_power.compressed(), 0.1)
        z_sensitivity = z_power_min + log_range + ma.mean(gas_atten, axis=0)
        zc = ma.median(ma.array(z, mask=~is_clutter), axis=0)
        z_sensitivity[~zc.mask] = zc[~zc.mask]
        dwell_time = utils.mdiff(time) * 3600  # seconds
        independent_pulses = (dwell_time * self.radar_frequency * 1e9 * 4
                              * np.sqrt(math.pi) * self.data['width'][:] / 3e8)
        z_precision = 4.343 * (1 / np.sqrt(independent_pulses)
                               + utils.db2lin(z_power_min - z_power) / 3)
        z_error = utils.l2norm(gas_atten * config.GAS_ATTEN_PREC, liq_atten_err,
                               z_precision)
        z_error[is_uncorrected_liquid] = ma.masked
        self._append_data(z_error, 'Z_error')
        self._append_data(z_sensitivity, 'Z_sensitivity')
        self._append_data(config.Z_BIAS, 'Z_bias')

    def add_meta(self):
        """Adds metadata to Radar() instance."""
        for key in ('latitude', 'longitude', 'altitude'):
            self._append_data(self._getvar(key), key)
        for key in ('time', 'height', 'radar_frequency'):
            self._append_data(getattr(self, key), key)


class Lidar(RawDataSource):
    """Class for lidar data.

    Attributes:
        height (ndarray): Altitude grid above mean sea level (m).

    """
    def __init__(self, lidar_file, fields):
        super().__init__(lidar_file)
        self.height = self._get_height()
        self.netcdf_to_cloudnet(fields)

    def rebin_data(self, time_new, height_new):
        """Rebins lidar data in time and height using mean."""
        for key in self.data:
            self.data[key].rebin_data(self.time, time_new, self.height,
                                      height_new)

    def add_meta(self):
        """Adds metadata to Lidar() instance."""
        self._append_data(self._getvar('wavelength'), 'lidar_wavelength')
        self._append_data(config.BETA_ERROR[0], 'beta_bias')
        self._append_data(config.BETA_ERROR[1], 'beta_error')


class Mwr(RawDataSource):
    """Class for microwave radiometer data."""
    def __init__(self, mwr_file):
        super().__init__(mwr_file)
        self._get_lwp_data()
        self._fix_time()

    def _get_lwp_data(self):
        key = utils.findkey(self.variables, ('LWP_data', 'lwp'))
        self._append_data(self.variables[key], 'lwp')
        self._append_data(self._calc_lwp_error(), 'lwp_error')

    def _calc_lwp_error(self):
        fractional_error, linear_error = config.LWP_ERROR
        return utils.l2norm(self.data['lwp'][:]*fractional_error, linear_error)

    def interpolate_to_cloudnet_grid(self, time):
        """Interpolates liquid water path to Cloudnet's dense time grid."""
        for key in self.data:
            fun = interp1d(self.time, self.data[key][:])
            self._append_data(fun(time), key)


class Model(RawDataSource):
    """Class for model data."""
    fields_dense = ('temperature', 'pressure', 'rh',
                    'gas_atten', 'specific_gas_atten',
                    'specific_saturated_gas_atten',
                    'specific_liquid_atten')
    fields_sparse = fields_dense + ('q', 'uwind', 'vwind')

    def __init__(self, model_file, alt_site):
        super().__init__(model_file)
        self.type = self._get_model_type()
        self.model_heights = self._get_model_heights(alt_site)
        self.mean_height = self._get_mean_height()
        self.netcdf_to_cloudnet(self.fields_sparse)
        self.data_sparse = {}
        self.data_dense = {}

    def _get_model_type(self):
        possible_keys = ('ecmwf', 'gdas')
        for key in possible_keys:
            if key in self.filename:
                return key
        return ''

    def _get_model_heights(self, alt_site):
        """Returns model heights for each time step."""
        return utils.km2m(self.variables['height']) + alt_site

    def _get_mean_height(self):
        return np.mean(np.array(self.model_heights), axis=0)

    def interpolate_to_common_height(self, wl_band, field_names):
        """Interpolates model variables to common height grid."""
        def _interpolate_variable():
            datai = np.zeros((len(self.time), len(self.mean_height)))
            for ind, (alt, prof) in enumerate(zip(self.model_heights, data)):
                fun = interp1d(alt, prof, fill_value='extrapolate')
                datai[ind, :] = fun(self.mean_height)
            return CloudnetArray(datai, key, units)

        for key in field_names:
            variable = self.variables[key]
            data = np.array(variable[:])
            units = variable.units
            if 'atten' in key:
                data = data[wl_band, :, :]
            self.data_sparse[key] = _interpolate_variable()
        self._append_data(self.time, 'model_time')
        self._append_data(self.mean_height, 'model_height')

    def interpolate_to_cloudnet_grid(self, field_names, time_new, height_new):
        """Interpolates model variables to Cloudnet's dense time / height grid."""
        for key in field_names:
            self.data_dense[key] = utils.interpolate_2d(self.time,
                                                        self.mean_height,
                                                        time_new, height_new,
                                                        self.data_sparse[key][:])

    def calc_wet_bulb(self):
        """Calculates wet-bulb temperature in dense grid."""
        wet_bulb_temp = atmos.wet_bulb(self.data_dense['temperature'],
                                       self.data_dense['pressure'],
                                       self.data_dense['rh'])
        self._append_data(wet_bulb_temp, 'Tw', units='K')

    def screen_fields(self):
        fields_to_keep = ('temperature', 'pressure', 'q', 'uwind', 'vwind')
        self.data_sparse = {key: self.data_sparse[key]
                            for key in fields_to_keep}


def generate_categorize(input_files, output_file, zlib=True):
    """ High level API to generate Cloudnet categorize file.

    """
    def _interpolate_to_cloudnet_grid():
        """ Interpolate variables to Cloudnet's dense grid."""
        model.interpolate_to_common_height(radar.wl_band, model.fields_sparse)
        model.interpolate_to_cloudnet_grid(model.fields_dense, *grid)
        model.calc_wet_bulb()
        mwr.interpolate_to_cloudnet_grid(final_time)
        radar.rebin_data(final_time)
        lidar.rebin_data(*grid)

    radar = Radar(input_files[0], ('Zh', 'v', 'ldr', 'width'))
    lidar = Lidar(input_files[1], ('beta',))
    mwr = Mwr(input_files[2])
    model = Model(input_files[3], radar.altitude)
    final_time, final_height = utils.time_grid(), radar.height
    grid = (final_time, final_height)
    _interpolate_to_cloudnet_grid()

    cbits, cbits_aux = classify.classify_measurements(radar.data['Zh'][:],
                                                      radar.data['v'][:],
                                                      radar.data['width'][:],
                                                      radar.data['ldr'][:],
                                                      lidar.data['beta'][:],
                                                      model.data['Tw'][:],
                                                      grid, model.type)

    gas_atten = atmos.gas_atten(model.data_dense,
                                cbits['category_bits'][:],
                                final_height)

    liq_atten, liq_atten_aux = atmos.liquid_atten(mwr.data,
                                                  model.data_dense,
                                                  cbits['category_bits'][:],
                                                  cbits_aux['liquid_bases'],
                                                  cbits_aux['is_rain'],
                                                  final_height)

    radar.correct_atten(gas_atten['radar_gas_atten'][:],
                        liq_atten['radar_liquid_atten'][:])

    quality_bits = classify.fetch_qual_bits(radar.data['Z'][:],
                                            lidar.data['beta'][:],
                                            cbits_aux['is_clutter'],
                                            liq_atten_aux)

    radar.calc_errors(gas_atten['radar_gas_atten'][:],
                      liq_atten_aux['liq_att_err'],
                      liq_atten_aux['is_not_corr'],
                      cbits_aux['is_clutter'],
                      final_time)
    radar.add_meta()
    lidar.add_meta()
    model.screen_fields()

    output_data = {**radar.data, **lidar.data, **model.data_sparse,
                   **mwr.data, **cbits, **gas_atten, **liq_atten,
                   **quality_bits}

    output.update_attributes(output_data)
    _save_cat(output_file, radar, model, output_data, zlib)


def _save_cat(file_name, radar, model, obs, zlib):
    """Creates a categorize netCDF4 file and saves all data into it."""
    rootgrp = netCDF4.Dataset(file_name, 'w', format='NETCDF4_CLASSIC')
    rootgrp.createDimension('time', len(radar.time))
    rootgrp.createDimension('height', len(radar.height))
    rootgrp.createDimension('model_time', len(model.time))
    rootgrp.createDimension('model_height', len(model.mean_height))
    output.write_vars2nc(rootgrp, obs, zlib)
    rootgrp.Conventions = 'CF-1.7'
    rootgrp.title = 'Categorize file from ' + radar.location
    rootgrp.institution = 'Data processed at the ' + config.INSTITUTE
    #dvec = radar_meta['date']
    #rootgrp.year = int(dvec[:4])
    #rootgrp.month = int(dvec[5:7])
    #rootgrp.day = int(dvec[8:])
    rootgrp.cloudnetpy_version = cloudnetpy.version.__version__
    #rootgrp.git_version = ncf.git_version()
    rootgrp.file_uuid = utils.get_uuid()
    rootgrp.references = 'https://doi.org/10.1175/BAMS-88-6-883'
    rootgrp.history = f"{utils.get_time()} - categorize file created"
    rootgrp.close()
