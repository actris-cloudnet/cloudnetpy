""" Functions for rebinning input data.
"""
import os
import math
import netCDF4
import numpy as np
import numpy.ma as ma
import scipy.constants
from scipy.interpolate import interp1d
from cloudnetpy import atmos, classify, output, utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.metadata import MetaData


def generate_categorize(input_files, output_file):
    """Generates Cloudnet categorize file.

    The measurements are rebinned into a common height / time grid,
    and classified as different types of scatterers such as ice, liquid,
    insects, etc. Next, the radar signal is corrected for atmospheric
    attenuation, and error estimates are computed. Results are saved
    in *ouput_file* which is by default a compressed NETCDF4_CLASSIC
    file.

    Args:
        input_files (dict): dict containing file names for calibrated
            'radar', 'lidar' 'model' and 'mwr' files.
        output_file (str): Full path of the output file.

    Notes:
        Separate mwr-file is not needed when using RPG cloud radar which
        measures liquid water path. Then, the radar file can be used as
        a mwr-file as well, i.e. {'mwr': 'radar.nc'}.

    Examples:
        >>> from cloudnetpy.categorize import generate_categorize
        >>> input_files = {'radar': 'radar.nc',
                           'lidar': 'lidar.nc',
                           'model': 'model.nc',
                           'mwr': 'mwr.nc'}
        >>> generate_categorize(input_files, 'output.nc')

    """

    def _interpolate_to_cloudnet_grid():
        model.interpolate_to_common_height(radar.wl_band)
        model.interpolate_to_grid(time, height)
        mwr.rebin_to_grid(time)
        radar.rebin_to_grid(time)
        lidar.rebin_to_grid(time, height)

    def _prepare_output():
        radar.add_meta()
        lidar.add_meta()
        model.screen_sparse_fields()
        for key in ('category_bits', 'insect_prob', 'is_rain'):
            radar.append_data(getattr(classification, key), key)
        for key in ('radar_liquid_atten', 'radar_gas_atten'):
            radar.append_data(attenuations[key], key)
        radar.append_data(quality['quality_bits'], 'quality_bits')
        return {**radar.data, **lidar.data, **model.data, **model.data_sparse,
                **mwr.data}

    def _define_dense_grid():
        return utils.time_grid(), radar.height

    radar = Radar(input_files['radar'])
    lidar = Lidar(input_files['lidar'])
    model = Model(input_files['model'], radar.altitude)
    mwr = Mwr(input_files['mwr'])
    time, height = _define_dense_grid()
    _interpolate_to_cloudnet_grid()
    model.calc_wet_bulb()
    classification = classify.classify_measurements(radar, lidar, model)
    attenuations = atmos.get_attenuations(model, mwr, classification)
    radar.correct_atten(attenuations)
    radar.calc_errors(attenuations, classification)
    quality = classify.fetch_quality(radar, lidar, classification, attenuations)
    output_data = _prepare_output()
    output.update_attributes(output_data, CATEGORIZE_ATTRIBUTES)
    _save_cat(output_file, radar, lidar, model, output_data)


def _save_cat(file_name, radar, lidar, model, obs):
    """Creates a categorize netCDF4 file and saves all data into it."""

    def _merge_source():
        # Probably should include mwr and model source if existing
        rootgrp.source = f"radar: {radar.source}\nlidar: {lidar.source}"

    dims = {'time': len(radar.time),
            'height': len(radar.height),
            'model_time': len(model.time),
            'model_height': len(model.mean_height)}
    rootgrp = output.init_file(file_name, dims, obs)
    output.copy_global(radar.dataset, rootgrp, ('year', 'month', 'day', 'location'))
    rootgrp.title = f"Categorize file from {radar.location}"
    # Needs to solve how to provide institution
    # rootgrp.institution = f"Data processed at {config.INSTITUTE}"
    rootgrp.references = 'https://doi.org/10.1175/BAMS-88-6-883'
    output.merge_history(rootgrp, 'categorize', radar, lidar)
    _merge_source()
    rootgrp.close()


class DataSource:
    """Base class for all Cloudnet measurements and model data.

    Args:
        filename (str): Calibrated instrument / model NetCDF file.

    Attributes:
        filename (str): Filename of the input file.
        dataset (Dataset): A netCDF4 Dataset instance.
        variables (dict): Variables of the Dataset instance.
        source (str): Global attribute 'source' from *input_file*.
        time (MaskedArray): Time array of the instrument.
        altitude (float): Altitude of instrument above mean sea level (m).
        data (dict): Dictionary containing CloudnetArray instances.

    """
    def __init__(self, filename):
        self.filename = os.path.basename(filename)
        self.dataset = netCDF4.Dataset(filename)
        self.variables = self.dataset.variables
        self.source = getattr(self.dataset, 'source', '')
        self.time = self._init_time()
        self.altitude = self._init_altitude()
        self.data = {}

    def _init_time(self):
        time = self.getvar('time')
        if max(time) > 24:
            time = utils.seconds2hours(time)
        return time

    def _init_altitude(self):
        """Returns altitude of the instrument (m)."""
        if 'altitude' in self.variables:
            altitude_above_sea = self.km2m(self.variables['altitude'])
            return float(altitude_above_sea)
        return None

    def getvar(self, *args):
        """Returns data array from the source file variables.

        Returns just the data (and no attributes) from the original variables
        dictionary, fetched from the input NetCDF file.

        Args:
            *args: possible names of the variable. The first match is returned.

        Returns:
            MaskedArray: The actual data.

        Raises:
             KeyError: The variable is not found.

        """
        for arg in args:
            if arg in self.variables:
                return self.variables[arg][:]
        raise KeyError('Missing variable in the input file.')

    def _netcdf_to_cloudnet(self, fields):
        """Transforms netCDF4-variables into CloudnetArrays.

        Args:
            fields (tuple): netCDF4-variables to be converted. The results are
                saved in *self.data* dictionary with *fields* strings as keys.

        Notes:
            The attributes of the variables are not copied. Just the data.

        """
        for key in fields:
            self.append_data(self.variables[key], key)

    def _unknown_to_cloudnet(self, possible_names, key, units=None):
        """Transforms single netCDF4 variable into CloudnetArray.

        Args:
            possible_names(tuple): Tuple of strings containing the possible
                names of the variable in the input NetCDF file.

            key(str): Key for self.data dictionary and name-attribute for
                the saved CloudnetArray object.

            units(str, optional): Units-attribute for the CloudnetArray object.

        """
        array = self.getvar(*possible_names)
        self.append_data(array, key, units=units)

    @staticmethod
    def km2m(var):
        """Converts km to m."""
        alt = var[:]
        if var.units == 'km':
            alt *= 1000
        return alt

    @staticmethod
    def m2km(var):
        """Converts m to km."""
        alt = var[:]
        if var.units == 'm':
            alt /= 1000
        return alt

    def append_data(self, data, key, name=None, units=None):
        """Adds new CloudnetVariable into self.data dictionary.

        Args:
            data (ndarray): Data to be added.
            key (str): Key for self.data dict.
            name (str, optional): CloudnetArray.name attribute. Default value
                is *key*.
            units (str, optional): CloudnetArray.units attribute.

        """
        self.data[key] = CloudnetArray(data, name or key, units)


class ProfileDataSource(DataSource):
    """ProfileDataSource class, child of DataSource.

    Args:
        filename (str): Raw lidar or radar file.

    Attributes:
        height (ndarray): Measurement height grid above mean sea level (m).

    """
    def __init__(self, filename):
        super().__init__(filename)
        self.height = self._get_height()

    def _get_height(self):
        """Returns height array above mean sea level (m)."""
        if 'height' in self.variables:
            return self.km2m(self.variables['height'])
        range_instrument = self.km2m(self.variables['range'])
        return np.array(range_instrument + self.altitude)


class Radar(ProfileDataSource):
    """Radar class, child of ProfileDataSource.

    Args:
        radar_file (str): File name of the calibrated radar file.

    Attributes:
        radar_frequency (float): Radar frequency (GHz).
        wl_band (int): Int corresponding to frequency 0 = 35.5 GHz, 1 = 94 GHz.
        folding_velocity (float): Radar's folding velocity (m/s).
        location (str): Location of the radar, copied from the global attribute
            'location' of the *radar_file*.

    """
    def __init__(self, radar_file):
        super().__init__(radar_file)
        self.radar_frequency = float(self.getvar('radar_frequency', 'frequency'))
        self.wl_band = utils.get_wl_band(self.radar_frequency)
        self.folding_velocity = self._get_folding_velocity()
        self.sequence_indices = self._get_sequence_indices()
        self.location = getattr(self.dataset, 'location', '')
        self._netcdf_to_cloudnet(('v', 'width', 'ldr'))
        self._unknown_to_cloudnet(('Zh', 'Zv', 'Ze'), 'Z', units='dBZ')
        self._init_sigma_v()

    def _init_sigma_v(self):
        """Initializes std of the velocity field. The std will be calculated
        later when rebinning the data."""
        self.append_data(self.getvar('v'), 'v_sigma')

    def _get_sequence_indices(self):
        """Mira has only one sequence and one folding velocity. RPG has
        several sequences with different folding velocities."""
        all_indices = np.arange(len(self.height))
        if not utils.isscalar(self.folding_velocity):
            starting_indices = self.getvar('chirp_start_indices')
            return np.split(all_indices, starting_indices[1:])
        return [all_indices]

    def _get_folding_velocity(self):
        for key in ('nyquist_velocity', 'NyquistVelocity'):
            if key in self.variables:
                return self.getvar(key)
        if 'prf' in self.variables:
            return float(self.getvar('prf') * scipy.constants.c
                         / (4 * self.radar_frequency * 1e9))
        raise KeyError('Unable to determine folding velocity')

    def rebin_to_grid(self, time_new):
        """Rebins radar data in time using mean.

        Args:
            time_new (ndarray): Target time array as fraction hour. Updates
                *time* attribute.

        """
        for key in self.data:
            if key == 'Z':
                self.data[key].db2lin()
                self.data[key].rebin_data(self.time, time_new)
                self.data[key].lin2db()
            elif key == 'v':
                # This has some problems with RPG data when folding is present.
                self.data[key].rebin_in_polar(self.time, time_new,
                                              self.folding_velocity,
                                              self.sequence_indices)
            elif key == 'v_sigma':
                self.data[key].calc_linear_std(self.time, time_new)
            else:
                self.data[key].rebin_data(self.time, time_new)
        self.time = time_new

    def correct_atten(self, attenuations):
        """Corrects radar echo for liquid and gas attenuation.

        Args:
            attenuations (dict): 2-D attenuations due to atmospheric gases.

        """
        z_corrected = self.data['Z'][:] + attenuations['radar_gas_atten']
        ind = ma.where(attenuations['radar_liquid_atten'])
        z_corrected[ind] += attenuations['radar_liquid_atten'][ind]
        self.append_data(z_corrected, 'Z')

    def calc_errors(self, attenuations, classification):
        """Calculates error and sensitivity of radar echo."""

        def _calc_sensitivity():
            """Returns sensitivity of radar as function of altitude."""
            mean_gas_atten = ma.mean(attenuations['radar_gas_atten'], axis=0)
            z_sensitivity = z_power_min + log_range + mean_gas_atten
            zc = ma.median(ma.array(z, mask=~classification.is_clutter), axis=0)
            z_sensitivity[~zc.mask] = zc[~zc.mask]
            return z_sensitivity

        def _calc_error():
            z_precision = 4.343 * (1 / np.sqrt(_number_of_pulses())
                                   + utils.db2lin(z_power_min - z_power) / 3)
            gas_error = attenuations['radar_gas_atten'] * 0.1
            liq_error = attenuations['liquid_atten_err'].filled(0)
            z_error = utils.l2norm(gas_error, liq_error, z_precision)
            z_error[attenuations['liquid_uncorrected']] = ma.masked
            return z_error

        def _number_of_pulses():
            """Returns number of independent pulses."""
            seconds_in_hour = 3600
            dwell_time = utils.mdiff(self.time) * seconds_in_hour
            return (dwell_time * self.radar_frequency * 1e9 * 4
                    * np.sqrt(math.pi) * self.data['width'][:] / 3e8)

        z = self.data['Z'][:]
        radar_range = self.km2m(self.variables['range'])
        log_range = utils.lin2db(radar_range, scale=20)
        z_power = z - log_range
        z_power_min = np.percentile(z_power.compressed(), 0.1)
        self.append_data(_calc_error(), 'Z_error')
        self.append_data(_calc_sensitivity(), 'Z_sensitivity')
        self.append_data(1, 'Z_bias')

    def add_meta(self):
        """Copies misc. metadata from the input file."""
        for key in ('latitude', 'longitude', 'altitude'):
            self.append_data(self.getvar(key), key)
        for key in ('time', 'height', 'radar_frequency'):
            self.append_data(getattr(self, key), key)


class Lidar(ProfileDataSource):
    """Lidar class, child of ProfileDataSource.

    Args:
        lidar_file (str): File name of the calibrated lidar file.
        fields (tuple, optional): Tuple of strings containing fields to be
            extracted from the *lidar_file*, e.g ('beta', 'beta_raw'). Default
            is ('beta',).

    Attributes:
        wavelength (float): Lidar wavelength (nm).

    """
    def __init__(self, lidar_file, fields=('beta',)):
        super().__init__(lidar_file)
        self._netcdf_to_cloudnet(fields)
        self.wavelength = float(self.getvar('wavelength'))

    def rebin_to_grid(self, time_new, height_new):
        """Rebins lidar data in time and height using mean.

        Args:
            time_new (ndarray): 1-D target time array (fraction hour).
            height_new (ndarray): 1-D target height array (m).

        """
        for key in self.data:
            self.data[key].rebin_data(self.time, time_new, self.height,
                                      height_new)

    def add_meta(self):
        """Copies misc. metadata from the input file."""
        self.append_data(self.wavelength, 'lidar_wavelength')
        self.append_data(0.5, 'beta_error')
        self.append_data(3, 'beta_bias')


class Mwr(DataSource):
    """Microwave radiometer class, child of DataSource.

    Args:
         mwr_file (str): File name of the calibrated mwr file.

    """
    def __init__(self, mwr_file):
        super().__init__(mwr_file)
        self._init_lwp_data()
        self._init_lwp_error()

    def _init_lwp_data(self):
        # TODO: How to deal with negative LWP values?
        lwp = self.getvar('LWP_data', 'lwp')
        lwp[lwp < 0] = 0
        self.append_data(lwp, 'lwp')

    def _init_lwp_error(self):
        # TODO: Check these error values
        random_error, bias = 0.25, 50
        lwp_error = utils.l2norm(self.data['lwp'][:]*random_error, bias)
        self.append_data(lwp_error, 'lwp_error', units='g m-2')

    def rebin_to_grid(self, time_grid):
        """Rebinning of lwp and its error."""
        for key in self.data:
            self.data[key].rebin_1d_data(self.time, time_grid)


class Model(DataSource):
    """Model class, child of DataSource.

    Args:
        model_file (str): File name of the NWP model file.
        alt_site (float): Altitude of the site above mean sea level (m).

    Attributes:
        type (str): Model type, e.g. 'gdas1' or 'ecwmf'.
        model_heights (ndarray): 2-D array of model heights (one for each time
            step).
        mean_height (ndarray): Mean of *model_heights*.
        data_sparse (dict): Model variables in common height grid but without
            interpolation in time.
        data_dense (dict): Model variables interpolated to Cloudnet's dense
            time / height grid.

    """
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
        self.height = None
        self.data_sparse = {}
        self.data_dense = {}
        self._append_grid()

    def _append_grid(self):
        self.append_data(self.time, 'model_time')
        self.append_data(self.mean_height, 'model_height')

    def _get_model_type(self):
        possible_keys = ('ecmwf', 'gdas')
        for key in possible_keys:
            if key in self.filename:
                return key
        return ''

    def _get_model_heights(self, alt_site):
        """Returns model heights for each time step."""
        model_heights = self.variables['height']
        if ma.count_masked(model_heights[:] > 0):
            raise RuntimeError('Masked values in the data file! Aborting..')
        return self.km2m(model_heights) + alt_site

    def _get_mean_height(self):
        return np.mean(np.array(self.model_heights), axis=0)

    def interpolate_to_common_height(self, wl_band):
        """Interpolates model variables to common height grid.

        Args:
            wl_band (int): Integer denoting the wavelength band of the
                radar (0=35.5 GHz, 1=94 GHz).

        """
        def _interpolate_variable():
            datai = np.zeros((len(self.time), len(self.mean_height)))
            for ind, (alt, prof) in enumerate(zip(self.model_heights, data)):
                fun = interp1d(alt, prof, fill_value='extrapolate')
                datai[ind, :] = fun(self.mean_height)
            return CloudnetArray(datai, key, units)

        for key in self.fields_sparse:
            variable = self.variables[key]
            data = np.array(variable[:])
            units = variable.units
            if 'atten' in key:
                data = data[wl_band, :, :]
            self.data_sparse[key] = _interpolate_variable()

    def interpolate_to_grid(self, time_grid, height_grid):
        """Interpolates model variables to Cloudnet's dense time / height grid.

        Args:
            time_grid (ndarray): The target time array (fraction hour).
            height_grid (ndarray): The target height array (m).

        """
        for key in self.fields_dense:
            self.data_dense[key] = utils.interpolate_2d(self.time,
                                                        self.mean_height,
                                                        self.data_sparse[key][:],
                                                        time_grid, height_grid)
        self.height = height_grid

    def calc_wet_bulb(self):
        """Calculates wet-bulb temperature in dense grid."""
        wet_bulb_temp = atmos.calc_wet_bulb_temperature(self.data_dense)
        self.append_data(wet_bulb_temp, 'Tw', units='K')

    def screen_sparse_fields(self):
        """Removes model fields that we don't want to write in the output."""
        fields_to_keep = ('temperature', 'pressure', 'q', 'uwind', 'vwind')
        self.data_sparse = {key: self.data_sparse[key]
                            for key in fields_to_keep}


COMMENTS = {
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
     'attribute. Bit 0 is the least significant.'),

    'LWP':
    ('This variable is the vertically integrated liquid water directly over the\n'
     'site. The temporal correlation of errors in liquid water path means that\n'
     'it is not really meaningful to distinguish bias from random error, so only\n'
     'an error variable is provided.'),

    'LWP_error':
    ('This variable is a rough estimate of the one-standard-deviation error\n'
     'in liquid water path, calculated as a combination of a 20 g m-2 linear\n'
     'error and a 25% fractional error.'),

    'radar_liquid_atten':
    ('This variable was calculated from the liquid water path measured by\n'
     'microwave radiometer using lidar and radar returns to perform an \n'
     'approximate partitioning of the liquid water content with height.\n'
     'Bit 5 of the quality_bits variable indicates where a correction for\n'
     'liquid water attenuation has been performed.'),

    'radar_gas_atten':
    ('This variable was calculated from the model temperature, pressure and\n'
     'humidity, but forcing pixels containing liquid cloud to saturation with\n'
     'respect to liquid water. It has been used to correct Z.'),

    'Tw':
    ('This variable was calculated from model T, P and relative humidity, first\n'
     'interpolated into measurement grid.'),

    'Z_sensitivity':
    ('This variable is an estimate of the radar sensitivity, i.e. the minimum\n'
     'detectable radar reflectivity, as a function of height. It includes the\n'
     'effect of ground clutter and gas attenuation but not liquid attenuation.'),

    'Z_error':
    ('This variable is an estimate of the one-standard-deviation random error in\n'
     'radar reflectivity factor. It originates from the following independent\n'
     'sources of error:\n'
     '1) Precision in reflectivity estimate due to finite signal to noise\n'
     '   and finite number of pulses\n'
     '2) 10% uncertainty in gaseous attenuation correction (mainly due to\n'
     '   error in model humidity field)\n'
     '3) Error in liquid water path (given by the variable lwp_error) and\n'
     '   its partitioning with height).'),

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
    'This variable is an estimate of the one-standard-deviation calibration error.',

}

DEFINITIONS = {
    'category_bits':
        ('\n'
         'Bit 0: Small liquid droplets are present.\n'
         'Bit 1: Falling hydrometeors are present; if Bit 2 is set then these are most\n'
         '       likely ice particles, otherwise they are drizzle or rain drops.\n'
         'Bit 2: Wet-bulb temperature is less than 0 degrees C, implying\n'
         '       the phase of Bit-1 particles.\n'
         'Bit 3: Melting ice particles are present.\n'
         'Bit 4: Aerosol particles are present and visible to the lidar.\n'
         'Bit 5: Insects are present and visible to the radar.'),

    'quality_bits':
        ('\n'
         'Bit 0: An echo is detected by the radar.\n'
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
         '       be aware that errors in reflectivity may result.'),

}

CATEGORIZE_ATTRIBUTES = {
    'Z': MetaData(
        long_name='Radar reflectivity factor',
        units='dBZ',
        comment=COMMENTS['Z'],
        ancillary_variables='Z_error Z_bias Z_sensitivity'
    ),
    'Z_error': MetaData(
        long_name='Error in radar reflectivity factor',
        units='dB',
        comment=COMMENTS['Z_error']
    ),
    'Z_bias': MetaData(
        long_name='Bias in radar reflectivity factor',
        units='dB',
        comment=COMMENTS['bias']
    ),
    'Z_sensitivity': MetaData(
        long_name='Minimum detectable radar reflectivity',
        units='dBZ',
        comment=COMMENTS['Z_sensitivity']
    ),
    'Zh': MetaData(
        long_name='Radar reflectivity factor (uncorrected), horizontal polarization',
        units='dBZ',
    ),
    'radar_liquid_atten': MetaData(
        long_name='Approximate two-way radar attenuation due to liquid water',
        units='dB',
        comment=COMMENTS['radar_liquid_atten']
    ),
    'radar_gas_atten': MetaData(
        long_name='Two-way radar attenuation due to atmospheric gases',
        units='dB',
        comment=COMMENTS['radar_gas_atten'],
        references='Liebe (1985, Radio Sci. 20(5), 1069-1089)'
    ),
    'Tw': MetaData(
        long_name='Wet-bulb temperature',
        units='K',
        comment=COMMENTS['Tw']
    ),
    'vwind': MetaData(
        long_name='Meridional wind',
        units='m s-1',
    ),
    'uwind': MetaData(
        long_name='Zonal wind',
        units='m s-1',
    ),
    'is_rain': MetaData(
        long_name='Presence of rain',
        comment='Integer denoting the rain (1) or no rain (0).'
    ),
    'beta': MetaData(
        long_name='Attenuated backscatter coefficient',
        units='sr-1 m-1',
        ancillary_variables='beta_error beta_bias'
    ),
    'beta_raw': MetaData(
        long_name='Raw attenuated backscatter coefficient',
        units='sr-1 m-1',
    ),
    'beta_error': MetaData(
        long_name='Error in attenuated backscatter coefficient',
        units='dB',
    ),
    'beta_bias': MetaData(
        long_name='Bias in attenuated backscatter coefficient',
        units='dB',
    ),
    'category_bits': MetaData(
        long_name='Target categorization bits',
        comment=COMMENTS['category_bits'],
        definition=DEFINITIONS['category_bits']
    ),
    'quality_bits': MetaData(
        long_name='Data quality bits',
        comment=COMMENTS['quality_bits'],
        definition=DEFINITIONS['quality_bits']
    ),
    'insect_prob': MetaData(
        long_name='Insect probability',
        units='',
    ),
    'lidar_wavelength': MetaData(
        long_name='Laser wavelength',
        units='nm'
    ),
}
