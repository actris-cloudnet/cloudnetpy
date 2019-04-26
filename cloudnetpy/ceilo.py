"""Module for reading and processing Vaisala ceilometers."""
import linecache
import numpy as np
import numpy.ma as ma
import scipy.ndimage
from cloudnetpy import utils, output
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.metadata import MetaData
import netCDF4


M2KM = 0.001
MINUTES_IN_HOUR = 60
SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 3600


class Ceilometer:
    """Base class for all types of ceilometers."""

    def __init__(self, file_name):
        self.file_name = file_name
        self.model = None
        self.backscatter = None
        self.metadata = None
        self.range = None
        self.time = None
        self.date = None
        self.noise_params = None
        self.data = {}

    def calc_beta(self):
        """Converts range-corrected raw beta to noise-screened beta."""

        def _screen_beta(beta_in, smooth):
            beta_in = self._calc_range_uncorrected_beta(beta_in, range_squared)
            beta_in = self._screen_by_snr(beta_in, is_saturation, smooth=smooth)
            return self._calc_range_corrected_beta(beta_in, range_squared)

        range_squared = self._get_range_squared()
        is_saturation = self._find_saturated_profiles()
        beta = _screen_beta(self.backscatter, False)
        # smoothed version:
        beta_smooth = ma.copy(self.backscatter)
        cloud_ind, cloud_values, cloud_limit = self._estimate_clouds_from_beta(beta)
        beta_smooth[cloud_ind] = cloud_limit
        sigma = self._calc_sigma_units()
        beta_smooth = scipy.ndimage.filters.gaussian_filter(beta_smooth, sigma)
        beta_smooth[cloud_ind] = cloud_values
        beta_smooth = _screen_beta(beta_smooth, True)
        return self.backscatter, beta, beta_smooth

    @staticmethod
    def _estimate_clouds_from_beta(beta):
        """Naively finds strong clouds from ceilometer backscatter."""
        cloud_limit = 1e-6
        cloud_ind = np.where(beta > cloud_limit)
        return cloud_ind, beta[cloud_ind], cloud_limit

    def _screen_by_snr(self, beta_uncorrected, is_saturation, smooth=False):
        """Screens noise from ceilometer backscatter.

        Args:
            beta_uncorrected (ndarray): Range-uncorrected backscatter.
            ceilo (obj): Ceilometer object.
            is_saturation (ndarray): Boolean array denoting saturated profiles.
            smooth (bool): Should be true if input beta is smoothed. Default is False.

        """
        beta = ma.copy(beta_uncorrected)
        n_gates, _, saturation_noise, noise_min = self.noise_params
        noise_min = noise_min[0] if smooth else noise_min[1]
        noise = self._estimate_noise_from_top_gates(beta, n_gates, noise_min)
        beta = self._reset_low_values_above_saturation(beta, is_saturation, saturation_noise)
        beta = self._remove_noise(beta, noise)
        return beta

    @staticmethod
    def _estimate_noise_from_top_gates(beta, n_gates, noise_min):
        """Estimates backscatter noise from topmost range gates."""
        noise = ma.std(beta[:, -n_gates:], axis=1)
        noise[noise < noise_min] = noise_min
        return noise

    @staticmethod
    def _reset_low_values_above_saturation(beta, is_saturation, saturation_noise):
        """Removes low values in saturated profiles above peak."""
        for saturated_profile in np.where(is_saturation)[0]:
            profile = beta[saturated_profile, :]
            peak_ind = np.argmax(profile)
            alt_ind = np.where(profile[peak_ind:] < saturation_noise)[0] + peak_ind
            beta[saturated_profile, alt_ind] = ma.masked
        return beta

    def _get_range_squared(self):
        """Returns range squared (km2)."""
        return (self.range*M2KM)**2

    @staticmethod
    def _remove_noise(beta, noise):
        """Removes points where snr < 5."""
        snr_limit = 5
        snr = (beta.T / noise)
        beta[snr.T < snr_limit] = ma.masked
        return beta

    @staticmethod
    def _calc_range_uncorrected_beta(beta, range_squared):
        return beta / range_squared

    @staticmethod
    def _calc_range_corrected_beta(beta, range_squared):
        return beta * range_squared

    def _find_saturated_profiles(self):
        """Estimates saturated profiles using the variance of the top range gates."""
        n_gates, var_lim, _, _ = self.noise_params
        var = np.var(self.backscatter[:, -n_gates:], axis=1)
        return var < var_lim

    def _calc_sigma_units(self):
        """Calculates Gaussian peak std parameters."""
        sigma_minutes = 2
        sigma_metres = 5
        time_step = utils.mdiff(self.time) * MINUTES_IN_HOUR
        alt_step = utils.mdiff(self.range)
        x = sigma_minutes / time_step
        y = sigma_metres / alt_step
        return x, y


class JenoptikCeilo(Ceilometer):
    """Class for Jenoptik ch15k ceilometer."""
    def __init__(self, file_name):
        super().__init__(file_name)
        self.model = 'ch15k'
        self.dataset = netCDF4.Dataset(self.file_name)
        self.variables = self.dataset.variables
        self.noise_params = (70, 2e-14, 0.3e-6, (1e-9, 4e-9))
        self.calibration_factor = 4.5e-11  # mace-head value

    def read_ceilometer_file(self):
        self.range = self._calc_range()
        self.time = self._convert_time()
        self.date = self._read_date()
        self.backscatter = self._convert_backscatter()
        self.metadata = self._read_metadata()

    def _read_date(self):
        """Read year, month, day from global attributes."""
        return [self.dataset.year, self.dataset.month, self.dataset.day]

    def _read_metadata(self):
        meta = {'tilt_angle': self._getvar('zenith')}
        return meta

    def _convert_backscatter(self):
        """Steps to convert (at least Mace Head) Jenoptik SNR to raw beta."""
        beta_raw = self._getvar('beta_raw')
        data_std = self._getvar('stddev')
        normalised_apd = self._get_nn()
        beta_raw *= utils.transpose(data_std / normalised_apd)
        beta_raw *= self.range**2
        beta_raw *= self.calibration_factor
        return beta_raw

    def _get_nn(self):
        """Taken from the Matlab code. Not sure what this is.."""
        nn1 = self._getvar('nn1')
        nn_reference = 140
        nn_step_factor = 1.24
        return nn_step_factor**(-(nn1-nn_reference)/5)

    def _calc_range(self):
        ceilo_range = self._getvar('range')
        return ceilo_range + utils.mdiff(ceilo_range)/2

    def _convert_time(self):
        time = self.variables['time']
        return utils.seconds2hours(time)

    def _getvar(self, name):
        """Reads data of variable (array or scalar) from netcdf-file."""
        var = self.variables[name]
        return var[0] if utils.isscalar(var) else var[:]


class VaisalaCeilo(Ceilometer):
    """Base class for Vaisala ceilometers."""
    def __init__(self, file_name):
        super().__init__(file_name)
        self._backscatter_scale_factor = None
        self._hex_conversion_params = None
        self._message_number = None

    def _fetch_data_lines(self):
        """Finds data lines (header + backscatter) from ceilometer file."""
        with open(self.file_name) as file:
            all_lines = file.readlines()
        return self._screen_empty_lines(all_lines)

    def _read_header_line_1(self, lines):
        """Reads all first header lines from CT25k and CL ceilometers."""
        fields = ('model_id', 'unit_id', 'software_level', 'message_number',
                  'message_subclass')
        if 'cl' in self.model:
            indices = [1, 3, 4, 7, 8, 9]
        else:
            indices = [1, 3, 4, 6, 7, 8]
        values = [_split_string(line, indices) for line in lines]
        return _values_to_dict(fields, values)

    def _calc_range(self):
        """Calculates range vector from the resolution and number of gates."""
        if self.model == 'ct25k':
            range_resolution = 30
            n_gates = 256
        else:
            n_gates = int(self.metadata['number_of_gates'])
            range_resolution = int(self.metadata['range_resolution'])
        return np.arange(n_gates)*range_resolution + range_resolution/2

    def _read_backscatter(self, lines):
        """Converts backscatter profile from 2-complement hex to floats."""
        n_chars = self._hex_conversion_params[0]
        n_gates = int(len(lines[0])/n_chars)
        profiles = np.zeros((len(lines), n_gates), dtype=int)
        ran = range(0, n_gates*n_chars, n_chars)
        for ind, line in enumerate(lines):
            try:
                profiles[ind, :] = [int(line[i:i+n_chars], 16) for i in ran]
            except ValueError as error:
                print(error)
        ind = np.where(profiles & self._hex_conversion_params[1] != 0)
        profiles[ind] -= self._hex_conversion_params[2]
        return profiles.astype(float) / self._backscatter_scale_factor

    @staticmethod
    def _screen_empty_lines(data):
        """Removes empty lines from the list of data."""

        def _parse_empty_lines():
            return [n for n, _ in enumerate(data) if is_empty_line(data[n])]

        def _parse_data_lines(empty_indices):
            number_of_data_lines = empty_indices[1] - empty_indices[0] - 1
            return [[data[n + line_number + 1] for n in empty_indices]
                    for line_number in range(number_of_data_lines)]

        empty_lines = _parse_empty_lines()
        return _parse_data_lines(empty_lines)

    @staticmethod
    def _read_header_line_2(lines):
        """Reads the second header line."""
        fields = ('detection_status', 'warning', 'cloud_base_data',
                  'warning_flags')
        values = [[line[0], line[1], line[3:20], line[21:].strip()] for line in lines]
        return _values_to_dict(fields, values)

    @staticmethod
    def _get_message_number(header_line_1):
        """Returns the message number."""
        msg_no = header_line_1['message_number']
        assert len(np.unique(msg_no)) == 1, 'Error: inconsistent message numbers.'
        return int(msg_no[0])

    @staticmethod
    def _calc_time(time_lines):
        """Returns the time vector as fraction hour."""
        time = [time_to_fraction_hour(line.split()[1]) for line in time_lines]
        return np.array(time)

    @staticmethod
    def _calc_date(time_lines):
        """Returns the date [yyyy, mm, dd]"""
        return time_lines[0].split()[0].strip('-').split('-')

    @classmethod
    def _handle_metadata(cls, header):
        meta = cls._concatenate_meta(header)
        meta = cls._remove_meta_duplicates(meta)
        meta = cls._convert_meta_strings(meta)
        return meta

    @staticmethod
    def _concatenate_meta(header):
        meta = {}
        for head in header:
            meta.update(head)
        return meta

    @staticmethod
    def _remove_meta_duplicates(meta):
        for field in meta:
            if len(np.unique(meta[field])) == 1:
                meta[field] = meta[field][0]
        return meta

    @staticmethod
    def _convert_meta_strings(meta):
        strings = ('cloud_base_data', 'measurement_parameters', 'cloud_amount_data')
        for field in meta:
            if field in strings:
                continue
            values = meta[field]
            if isinstance(values, str):  # only one unique value
                try:
                    meta[field] = int(values)
                except (ValueError, TypeError):
                    continue
            else:
                meta[field] = [None] * len(values)
                for ind, value in enumerate(values):
                    try:
                        meta[field][ind] = int(value)
                    except (ValueError, TypeError):
                        continue
                meta[field] = np.array(meta[field])
        return meta

    def _read_header_line_3(self, data):
        raise NotImplementedError

    def _read_common_header_part(self):
        header = []
        data_lines = self._fetch_data_lines()
        self.time = self._calc_time(data_lines[0])
        self.date = self._calc_date(data_lines[0])
        header.append(self._read_header_line_1(data_lines[1]))
        self._message_number = self._get_message_number(header[0])
        header.append(self._read_header_line_2(data_lines[2]))
        header.append(self._read_header_line_3(data_lines[3]))
        return header, data_lines

    def _range_correct_upper_part(self):
        """Range corrects the upper part of profile."""
        altitude_limit = 2400
        ind = np.where(self.range > altitude_limit)
        self.backscatter[:, ind] *= (self.range[ind]*M2KM)**2


class ClCeilo(VaisalaCeilo):
    """Base class for Vaisala CL31/CL51 ceilometers."""

    def __init__(self, file_name):
        super().__init__(file_name)
        self._hex_conversion_params = (5, 524288, 1048576)
        self._backscatter_scale_factor = 1e8
        self.noise_params = (100, 1e-12, 3e-6, (1.1e-8, 2.9e-8))

    def read_ceilometer_file(self):
        """Read all lines of data from the file."""
        header, data_lines = self._read_common_header_part()
        header.append(self._read_header_line_4(data_lines[-3]))
        self.metadata = self._handle_metadata(header)
        self.range = self._calc_range()
        self.backscatter = self._read_backscatter(data_lines[-2])

    def _read_header_line_3(self, lines):
        if self._message_number != 2:
            return None
        keys = ('cloud_detection_status', 'cloud_amount_data')
        values = [[line[0:3], line[3:].strip()] for line in lines]
        return _values_to_dict(keys, values)

    @staticmethod
    def _read_header_line_4(lines):
        keys = ('scale', 'range_resolution', 'number_of_gates', 'laser_energy',
                'laser_temperature', 'window_transmission', 'tilt_angle',
                'background_light', 'measurement_parameters', 'backscatter_sum')
        values = [line.split() for line in lines]
        return _values_to_dict(keys, values)


class Cl51(ClCeilo):
    """Class for Vaisala CL51 ceilometer."""
    def __init__(self, input_file):
        super().__init__(input_file)
        self.model = 'cl51'


class Cl31(ClCeilo):
    """Class for Vaisala CL31 ceilometer."""
    def __init__(self, input_file):
        super().__init__(input_file)
        self.model = 'cl31'


class Ct25k(VaisalaCeilo):
    """Class for Vaisala CT25k ceilometer.

    References:
        https://www.manualslib.com/manual/1414094/Vaisala-Ct25k.html

    """
    def __init__(self, input_file):
        super().__init__(input_file)
        self.model = 'ct25k'
        self._hex_conversion_params = (4, 32768, 65536)
        self._backscatter_scale_factor = 1e7
        self.noise_params = (40, 2e-14, 0.3e-6, (3e-10, 1.5e-9))

    def read_ceilometer_file(self):
        """Read all lines of data from the file."""
        header, data_lines = self._read_common_header_part()
        self.metadata = self._handle_metadata(header)
        self.range = self._calc_range()
        hex_profiles = self._parse_hex_profiles(data_lines[4:20])
        self.backscatter = self._read_backscatter(hex_profiles)
        # should study the background noise to determine if the
        # next call is needed. It can be the case with cl31/51 also.
        self._range_correct_upper_part()

    @staticmethod
    def _parse_hex_profiles(lines):
        """Collects ct25k profiles into list (one profile / element)."""
        n_profiles = len(lines[0])
        return [''.join([lines[l][n][3:].strip() for l in range(16)])
                for n in range(n_profiles)]

    def _read_header_line_3(self, lines):
        if self._message_number in (1, 3, 6):
            return None
        keys = ('scale', 'measurement_mode', 'laser_energy',
                'laser_temperature', 'receiver_sensitivity',
                'window_contamination', 'tilt_angle', 'background_light',
                'measurement_parameters', 'backscatter_sum')
        values = [line.split() for line in lines]
        return _values_to_dict(keys, values)


def ceilo2nc(input_file, output_file, location='unknown', altitude=0):
    """Converts Vaisala and Jenoptik raw files into netCDF.

    Args:
        input_file (str): Ceilometer file name. For Vaisala it is a text file,
            for Jenoptik it is a netCDF file.
        output_file (str): Output file name, e.g. 'ceilo.nc'.
        location (str, optional): Name of the measurement site, e.g. 'Kumpula'.
            Default is 'unknown'.
        altitude (int, optional): Altitude of the instrument above
            mean sea level (m). Default is 0.

    """
    ceilo = _initialize_ceilo(input_file)
    ceilo.read_ceilometer_file()
    beta_variants = ceilo.calc_beta()
    _append_data(ceilo, beta_variants)
    _append_height(ceilo, altitude)
    output.update_attributes(ceilo.data, ATTRIBUTES)
    _save_ceilo(ceilo, output_file, location)


def _append_height(ceilo, site_altitude):
    """Finds height above mean sea level."""
    tilt_angle = np.median(ceilo.metadata['tilt_angle'])
    height = _calc_height(ceilo.range, tilt_angle)
    height += site_altitude
    ceilo.data['height'] = CloudnetArray(height, 'height')


def _calc_height(ceilo_range, tilt_angle):
    """Calculates height from range and tilt angle."""
    return ceilo_range * np.cos(np.deg2rad(tilt_angle))


def _append_data(ceilo, beta_variants):
    """Add data and metadata as CloudnetArray's to ceilo.data attribute."""
    for data, name in zip(beta_variants, ('beta_raw', 'beta', 'beta_smooth')):
        ceilo.data[name] = CloudnetArray(data, name)
    for field in ('range', 'time'):
        ceilo.data[field] = CloudnetArray(getattr(ceilo, field), field)
    for field, data in ceilo.metadata.items():
        first_element = data if utils.isscalar(data) else data[0]
        if not isinstance(first_element, str):  # String array writing not yet supported
            ceilo.data[field] = CloudnetArray(np.array(ceilo.metadata[field],
                                                       dtype=float), field)


def _save_ceilo(ceilo, output_file, location):
    """Saves the ceilometer netcdf-file."""
    dims = {'time': len(ceilo.time), 'range': len(ceilo.range)}
    rootgrp = output.init_file(output_file, dims, ceilo.data, zlib=True)
    rootgrp.title = f"Ceilometer file from {location}"
    rootgrp.year, rootgrp.month, rootgrp.day = ceilo.date
    rootgrp.location = location
    rootgrp.history = f"{utils.get_time()} - ceilometer file created"
    rootgrp.source = ceilo.model
    rootgrp.close()


def _values_to_dict(keys, values):
    out = {}
    for i, key in enumerate(keys):
        out[key] = np.array([x[i] for x in values])
    return out


def _split_string(string, indices):
    """Split string between indices."""
    return [string[n:m] for n, m in zip(indices[:-1], indices[1:])]


def _initialize_ceilo(file):
    model = _find_ceilo_model(file)
    if model == 'cl51':
        return Cl51(file)
    elif model == 'cl31':
        return Cl31(file)
    elif model == 'ct25k':
        return Ct25k(file)
    elif model == 'ch15k':
        return JenoptikCeilo(file)
    else:
        raise SystemExit('Error: Unknown ceilo model.')


def _find_ceilo_model(file):
    if file.endswith('nc'):
        return 'ch15k'
    first_empty_line = _find_first_empty_line(file)
    hint = linecache.getline(file, first_empty_line + 2)[1:5]
    if hint == 'CL01':
        return 'cl51'
    elif hint == 'CL02':
        return 'cl31'
    elif hint == 'CT02':
        return 'ct25k'
    return None


def _find_first_empty_line(file_name):
    line_number = 1
    with open(file_name) as file:
        for line in file:
            if is_empty_line(line):
                break
            line_number += 1
    return line_number


def is_empty_line(line):
    """Tests if line in text file is empty."""
    if line in ('\n', '\r\n'):
        return True
    return False


def time_to_fraction_hour(time):
    """ Time (hh:mm:ss) as fraction hour """
    h, m, s = time.split(':')
    return int(h) + (int(m) * SECONDS_IN_MINUTE + int(s)) / SECONDS_IN_HOUR


ATTRIBUTES = {
    'beta': MetaData(
        long_name='Attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment='Range corrected, SNR screened, attenuated backscatter.'
    ),
    'beta_raw': MetaData(
        long_name='Raw attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment="Range corrected, attenuated backscatter."
    ),
    'beta_smooth': MetaData(
        long_name='Smoothed attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment=('Range corrected, SNR screened backscatter coefficient.\n'
                 'Weak background is smoothed using Gaussian 2D-kernel.')
    ),
    'scale': MetaData(
        long_name='Scale',
        units='%',
        comment='100 (%) is normal.'
    ),
    'software_level': MetaData(
        long_name='Software level ID',
        units='',
    ),
    'laser_temperature': MetaData(
        long_name='Laser temperature',
        units='C',
    ),
    'window_transmission': MetaData(
        long_name='Window transmission estimate',
        units='%',
    ),
    'tilt_angle': MetaData(
        long_name='Tilt angle from vertical',
        units='degrees',
    ),
    'laser_energy': MetaData(
        long_name='Laser pulse energy',
        units='%',
    ),
    'background_light': MetaData(
        long_name='Background light',
        units='mV',
        comment='Measured at internal ADC input.'
    ),
    'backscatter_sum': MetaData(
        long_name='Sum of detected and normalized backscatter',
        units='sr-1',
        comment='Multiplied by scaling factor times 1e4.',
    ),
    'range_resolution': MetaData(
        long_name='Range resolution',
        units='m',
    ),
    'number_of_gates': MetaData(
        long_name='Number of range gates in profile',
        units='',
    ),
    'unit_id': MetaData(
        long_name='Ceilometer unit number',
        units='',
    ),
    'message_number': MetaData(
        long_name='Message number',
        units='',
    ),
    'message_subclass': MetaData(
        long_name='Message subclass number',
        units='',
    ),
    'detection_status': MetaData(
        long_name='Detection status',
        units='',
        comment='From the internal software of the instrument.'
    ),
    'warning': MetaData(
        long_name='Warning and Alarm flag',
        units='',
        definition=('\n'
                    'Value 0: Self-check OK\n'
                    'Value W: At least one warning on\n'
                    'Value A: At least one error active.')
    ),
    'warning_flags': MetaData(
        long_name='Warning flags',
        units='',
    ),
    'receiver_sensitivity': MetaData(
        long_name='Receiver sensitivity',
        units='%',
        comment='Expressed as % of nominal factory setting.'
    ),
    'window_contamination': MetaData(
        long_name='Window contamination',
        units='mV',
        comment='Measured at internal ADC input.'
    )
}
