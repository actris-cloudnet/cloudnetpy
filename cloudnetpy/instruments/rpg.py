"""This module contains RPG Cloud Radar related functions."""
from collections import namedtuple
from typing import Union, Tuple, Optional
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils, output, CloudnetArray
from cloudnetpy.metadata import MetaData


def rpg2nc(path_to_l1_files: str,
           output_file: str,
           site_meta: dict,
           keep_uuid: Optional[bool] = False,
           uuid: Optional[str] = None,
           date: Optional[str] = None) -> Tuple[str, list]:
    """Converts RPG FMCW-94 cloud radar data into Cloudnet Level 1b netCDF file.

    This function reads one day of RPG Level 1 cloud radar binary files,
    concatenates the data and writes it into netCDF file.

    Args:
        path_to_l1_files: Folder containing one day of RPG LV1 files.
        output_file: Output file name.
        site_meta: Dictionary containing information about the
            site. Required key value pairs are `altitude` (metres above mean
            sea level) and `name`.
        keep_uuid: If True, keeps the UUID of the old file,
            if that exists. Default is False when new UUID is generated.
        uuid: Set specific UUID for the file.
        date: Expected date in the input files. If not set,
            all files will be used. This might cause unexpected behavior if
            there are files from several days. If date is set as 'YYYY-MM-DD',
            only files that match the date will be used.

    Returns:
        2-element tuple containing

        - UUID of the generated file.
        - Files used in the processing.

    Raises:
        RuntimeError: Failed to read the binary data.

    Examples:
        >>> from cloudnetpy.instruments import rpg2nc
        >>> site_meta = {'name': 'Hyytiala', 'altitude': 174}
        >>> rpg2nc('/path/to/files/', 'test.nc', site_meta)

    """
    l1_files = utils.get_sorted_filenames(path_to_l1_files, '.LV1')
    rpg_objects, valid_files = get_rpg_objects(l1_files, date)
    one_day_of_data = create_one_day_data_record(rpg_objects)
    if not valid_files:
        return '', []
    rpg = Rpg(one_day_of_data, site_meta, 'RPG-FMCW-94')
    rpg.mask_invalid_ldr()
    rpg.linear_to_db(('Ze', 'antenna_gain'))
    attributes = output.add_time_attribute(RPG_ATTRIBUTES, rpg.date)
    output.update_attributes(rpg.data, attributes)
    return save_rpg(rpg, output_file, valid_files, keep_uuid, uuid)


def create_one_day_data_record(rpg_objects: list) -> dict:
    """Concatenates all RPG data from one day."""
    rpg_raw_data, rpg_header = stack_rpg_data(rpg_objects)
    if len(rpg_objects) > 1:
        try:
            rpg_header = reduce_header(rpg_header)
        except AssertionError as error:
            raise RuntimeError(error)
    rpg_raw_data = mask_invalid_data(rpg_raw_data)
    return {**rpg_header, **rpg_raw_data}


def get_rpg_objects(files: list, expected_date: Union[str, None]) -> Tuple[list, list]:
    """Creates a list of Rpg() objects from the file names."""
    objects = []
    valid_files = []
    for file in files:
        try:
            obj = RpgBin(file)
            if expected_date is not None:
                _validate_date(obj, expected_date)
        except (TypeError, ValueError) as err:
            print(err)
            continue
        objects.append(obj)
        valid_files.append(file)
    return objects, valid_files


def _validate_date(obj, expected_date: Union[str, None]) -> None:
    for t in obj.data['time'][:]:
        date_str = '-'.join(utils.seconds2date(t)[:3])
        if date_str != expected_date:
            raise ValueError('Warning: Ignoring a file (time stamps not what expected)')


def stack_rpg_data(rpg_objects):
    """Combines data from hourly Rpg() objects.

    Notes:
        Ignores variable names starting with an underscore.

    """
    def _stack(source, target, fun):
        for name, value in source.items():
            if not name.startswith('_'):
                target[name] = (fun((target[name], value)) if name in target else value)
    data, header = {}, {}
    for rpg in rpg_objects:
        _stack(rpg.data, data, np.concatenate)
        _stack(rpg.header, header, np.vstack)
    return data, header


def reduce_header(header_in: dict) -> dict:
    """Removes duplicate header data."""
    header = header_in.copy()
    for name in header:
        first_row = header[name][0]
        assert np.isclose(header[name], first_row, rtol=1e-3).all(), f"Inconsistent header: {name}"
        header[name] = first_row
    return header


def mask_invalid_data(rpg_data):
    for name in rpg_data:
        rpg_data[name] = ma.masked_equal(rpg_data[name], 0)
    return rpg_data


class Rpg:
    """Class for RPG FMCW-94 and HATPRO data."""
    def __init__(self, raw_data: dict, site_properties: dict, source: str):
        self.raw_data = raw_data
        self.date = self._get_date()
        self.raw_data['time'] = utils.seconds2hours(self.raw_data['time'])
        self.raw_data['altitude'] = np.array(site_properties['altitude'])
        self.data = self._init_data()
        self.source = source
        self.location = site_properties['name']

    def mask_invalid_ldr(self) -> None:
        self.data['ldr'].data = ma.masked_less_equal(self.data['ldr'].data, -35)

    def linear_to_db(self, variables_to_log: tuple) -> None:
        """Changes some linear units to logarithmic."""
        for name in variables_to_log:
            self.data[name].lin2db()

    def _init_data(self) -> dict:
        data = {}
        for key in self.raw_data:
            data[key] = CloudnetArray(self.raw_data[key], key)
        return data

    def _get_date(self) -> list:
        time_first = self.raw_data['time'][0]
        time_last = self.raw_data['time'][-1]
        date_first = utils.seconds2date(time_first)[:3]
        date_last = utils.seconds2date(time_last)[:3]
        if date_first != date_last:
            print('Warning: Measurements from different days')
        return date_first


def save_rpg(rpg: Rpg,
             output_file: str,
             valid_files: list,
             keep_uuid: bool,
             uuid: Union[str, None] = None) -> Tuple[str, list]:
    """Saves the RPG radar / mwr file."""

    dims = {'time': len(rpg.data['time'][:])}

    if 'fmcw' in rpg.source.lower():
        dims['range'] = len(rpg.data['range'][:])
        dims['chirp_sequence'] = len(rpg.data['chirp_start_indices'][:])
        file_type = 'radar'
    else:
        file_type = 'mwr'

    rootgrp = output.init_file(output_file, dims, rpg.data, keep_uuid, uuid)
    file_uuid = rootgrp.file_uuid
    output.add_file_type(rootgrp, file_type)
    rootgrp.title = f"{file_type.capitalize()} file from {rpg.location}"
    rootgrp.year, rootgrp.month, rootgrp.day = rpg.date
    rootgrp.location = rpg.location
    rootgrp.history = f"{utils.get_time()} - {file_type} file created"
    rootgrp.source = rpg.source
    output.add_references(rootgrp)
    rootgrp.close()
    return file_uuid, valid_files


class RpgBin:
    """RPG Cloud Radar Level 1 data reader."""
    def __init__(self, filename):
        self.filename = filename
        self._file_position = 0
        self.header = self.read_rpg_header()
        self.data = self.read_rpg_data()

    def read_rpg_header(self):
        """Reads the header or rpg binary file."""

        def append(names, dtype=np.int32, n_values=1):
            """Updates header dictionary."""
            for name in names:
                header[name] = np.fromfile(file, dtype, int(n_values))

        header = {}
        file = open(self.filename, 'rb')
        append(('file_code',
                '_header_length'), np.int32)
        append(('_start_time',
                '_stop_time'), np.uint32)
        append(('program_number',))
        append(('model_number',))  # 0 = single polarization, 1 = dual pol.
        header['_program_name'] = self.read_string(file)
        header['_customer_name'] = self.read_string(file)
        append(('radar_frequency',
                'antenna_separation',
                'antenna_diameter',
                'antenna_gain',  # linear
                'half_power_beam_width'), np.float32)
        append(('dual_polarization',), np.int8)  # 0 = single pol, 1 = LDR, 2 = STSR
        append(('sample_duration',), np.float32)
        append(('latitude',
                'longitude'), np.float32)
        append(('calibration_interval',
                '_number_of_range_gates',
                '_number_of_temperature_levels',
                '_number_of_humidity_levels',
                '_number_of_chirp_sequences'))
        append(('range',), np.float32, int(header['_number_of_range_gates']))
        append(('_temperature_levels',), np.float32,
               int(header['_number_of_temperature_levels']))
        append(('_humidity_levels',), np.float32,
               int(header['_number_of_humidity_levels']))
        append(('number_of_spectral_samples',
                'chirp_start_indices',
                'number_of_averaged_chirps'),
               n_values=int(header['_number_of_chirp_sequences']))
        append(('integration_time',
                'range_resolution',
                'nyquist_velocity'), np.float32,
               int(header['_number_of_chirp_sequences']))
        append(('_is_power_levelling',
                '_is_spike_filter',
                '_is_phase_correction',
                '_is_relative_power_correction'), np.int8)
        append(('FFT_window',), np.int8)  # 0=square, 1=parzen, 2=blackman, 3=welch, 4=slepian2, 5=slepian3
        append(('input_voltage_range',))
        append(('noise_threshold',), np.float32)
        # Fix for Level 1 version 4 files:
        if int(header['file_code']) >= 889348:
            _ = np.fromfile(file, np.int32, 25)
            _ = np.fromfile(file, np.uint32, 10000)
        self._file_position = file.tell()
        file.close()
        return header

    @staticmethod
    def read_string(file_id):
        """Read characters from binary data until whitespace."""
        str_out = ''
        while True:
            c = np.fromfile(file_id, np.int8, 1)
            if len(c) == 0 or c[0] == 0:
                break
            str_out += chr(c[0])
        return str_out

    def read_rpg_data(self):
        """Reads the actual data from rpg binary file."""
        Dimensions = namedtuple('Dimensions', ['n_samples',
                                               'n_gates',
                                               'n_layers_t',
                                               'n_layers_h'])

        def _create_dimensions():
            """Returns possible lengths of the data arrays."""
            n_samples = np.fromfile(file, np.int32, 1)
            return Dimensions(int(n_samples),
                              int(self.header['_number_of_range_gates']),
                              int(self.header['_number_of_temperature_levels']),
                              int(self.header['_number_of_humidity_levels']))

        def _create_variables():
            """Initializes dictionaries for data arrays."""
            vrs = {'sample_length': np.zeros(dims.n_samples, int),
                   'time': np.zeros(dims.n_samples, int),
                   'time_ms': np.zeros(dims.n_samples, int),
                   'quality_flag': np.zeros(dims.n_samples, int)}

            block1_vars = dict.fromkeys((
                'rain_rate',
                'relative_humidity',
                'temperature',
                'pressure',
                'wind_speed',
                'wind_direction',
                'voltage',
                'brightness_temperature',
                'lwp',
                'if_power',
                'elevation',
                'azimuth',
                'status_flag',
                'transmitted_power',
                'transmitter_temperature',
                'receiver_temperature',
                'pc_temperature'))

            block2_vars = dict.fromkeys((  # vertical polarization
                'Ze',
                'v',
                'width',
                'skewness',
                '_kurtosis'))

            if self.header['dual_polarization'] == 1:
                block2_vars.update(dict.fromkeys((
                    'ldr',
                    'correlation_coefficient',
                    'spectral_differential_phase')))
            elif self.header['dual_polarization'] == 2:
                block2_vars.update(dict.fromkeys((
                    'Zdr'
                    'correlation_coefficient'
                    'spectral_differential_phase'
                    '_',
                    'spectral_slanted_ldr',
                    'spectral_slanted_correlation_coefficient',
                    'specific_differential_phase_shift',
                    'differential_attenuation')))
            return vrs, block1_vars, block2_vars

        def _add_sensitivities():
            ind0 = len(block1) + n_dummy
            ind1 = ind0 + dims.n_gates
            block1['_sensitivity_limit_v'] = float_block1[:, ind0:ind1]
            if self.header['dual_polarization']:
                block1['_sensitivity_limit_h'] = float_block1[:, ind1:]

        def _get_length_of_dummy_data():
            return 3 + dims.n_layers_t + 2*dims.n_layers_h

        def _get_length_of_sensitivity_data():
            if self.header['dual_polarization']:
                return 2*dims.n_gates
            return dims.n_gates

        def _get_float_block_lengths():
            block_one_length = len(block1) + n_dummy + n_sens
            block_two_length = len(block2)
            return block_one_length, block_two_length

        def _init_float_blocks():
            block_one = np.zeros((dims.n_samples, n_floats1))
            block_two = np.zeros((dims.n_samples, dims.n_gates, n_floats2))
            return block_one, block_two

        file = open(self.filename, 'rb')
        file.seek(self._file_position)
        dims = _create_dimensions()
        aux, block1, block2 = _create_variables()
        n_dummy = _get_length_of_dummy_data()
        n_sens = _get_length_of_sensitivity_data()
        n_floats1, n_floats2 = _get_float_block_lengths()
        float_block1, float_block2 = _init_float_blocks()

        for sample in range(dims.n_samples):
            aux['sample_length'][sample] = np.fromfile(file, np.int32, 1)
            aux['time'][sample] = np.fromfile(file, np.uint32, 1)
            aux['time_ms'][sample] = np.fromfile(file, np.int32, 1)
            aux['quality_flag'][sample] = np.fromfile(file, np.int8, 1)
            float_block1[sample, :] = np.fromfile(file, np.float32, n_floats1)
            is_data = np.fromfile(file, np.int8, dims.n_gates)
            is_data_ind = np.where(is_data)[0]
            n_valid = len(is_data_ind)
            values = np.fromfile(file, np.float32, n_floats2*n_valid)
            float_block2[sample, is_data_ind, :] = values.reshape(n_valid, n_floats2)
        file.close()
        for n, name in enumerate(block1):
            block1[name] = float_block1[:, n]
        _add_sensitivities()
        for n, name in enumerate(block2):
            block2[name] = float_block2[:, :, n]
        return {**aux, **block1, **block2}


DEFINITIONS = {
    'model_number':
        ('\n'
         '0: Single polarisation radar.\n'
         '1: Dual polarisation radar.'),

    'dual_polarization':
        ('\n'
         'Value 0: Single polarisation radar.\n'
         'Value 1: Dual polarisation radar in linear depolarisation ratio (LDR)\n'
         '         mode.\n'
         'Value 2: Dual polarisation radar in simultaneous transmission\n'
         '         simultaneous reception (STSR) mode.'),

    'FFT_window':
        ('\n'
         'Value 0: Square\n'
         'Value 1: Parzen\n'
         'Value 2: Blackman\n'
         'Value 3: Welch\n'
         'Value 4: Slepian2\n'
         'Value 5: Slepian3'),

    'quality_flag':
        ('\n'
         'Bit 0: ADC saturation.\n'
         'Bit 1: Spectral width too high.\n'
         'Bit 2: No transmission power levelling.')

}

RPG_ATTRIBUTES = {
    'file_code': MetaData(
        long_name='File code',
        comment='Indicates the RPG software version.',
    ),
    'program_number': MetaData(
        long_name='Program number',
    ),
    'model_number': MetaData(
        long_name='Model number',
        definition=DEFINITIONS['model_number']
    ),
    'antenna_separation': MetaData(
        long_name='Antenna separation',
        units='m',
    ),
    'antenna_diameter': MetaData(
        long_name='Antenna diameter',
        units='m',
    ),
    'antenna_gain': MetaData(
        long_name='Antenna gain',
        units='dB',
    ),
    'half_power_beam_width': MetaData(
        long_name='Half power beam width',
        units='degrees',
    ),
    'dual_polarization': MetaData(
        long_name='Dual polarisation type',
        definition=DEFINITIONS['dual_polarization']
    ),
    'sample_duration': MetaData(
        long_name='Sample duration',
        units='s'
    ),
    'calibration_interval': MetaData(
        long_name='Calibration interval in samples'
    ),
    'number_of_spectral_samples': MetaData(
        long_name='Number of spectral samples in each chirp sequence',
        units='',
    ),
    'chirp_start_indices': MetaData(
        long_name='Chirp sequences start indices'
    ),
    'number_of_averaged_chirps': MetaData(
        long_name='Number of averaged chirps in sequence'
    ),
    'integration_time': MetaData(
        long_name='Integration time',
        units='s',
        comment='Effective integration time of chirp sequence',
    ),
    'range_resolution': MetaData(
        long_name='Vertical resolution of range',
        units='m',
    ),
    'FFT_window': MetaData(
        long_name='FFT window type',
        definition=DEFINITIONS['FFT_window']
    ),
    'input_voltage_range': MetaData(
        long_name='ADC input voltage range (+/-)',
        units='mV',
    ),
    'noise_threshold': MetaData(
        long_name='Noise filter threshold factor',
        units='',
        comment='Multiple of the standard deviation of Doppler spectra.'
    ),
    'time_ms': MetaData(
        long_name='Time ms',
        units='ms',
    ),
    'quality_flag': MetaData(
        long_name='Quality flag',
        definition=DEFINITIONS['quality_flag']
    ),
    'voltage': MetaData(
        long_name='Voltage',
        units='V',
    ),
    'brightness_temperature': MetaData(
        long_name='Brightness temperature',
        units='K',
    ),
    'if_power': MetaData(
        long_name='IF power at ACD',
        units='uW',
    ),
    'elevation': MetaData(
        long_name='Elevation angle above horizon',
        units='degrees',
    ),
    'azimuth': MetaData(
        long_name='Azimuth angle',
        units='degrees',
    ),
    'status_flag': MetaData(
        long_name='Status flag for heater and blower'
    ),
    'transmitted_power': MetaData(
        long_name='Transmitted power',
        units='W',
    ),
    'transmitter_temperature': MetaData(
        long_name='Transmitter temperature',
        units='K',
    ),
    'receiver_temperature': MetaData(
        long_name='Receiver temperature',
        units='K',
    ),
    'pc_temperature': MetaData(
        long_name='PC temperature',
        units='K',
    ),
    'skewness': MetaData(
        long_name='Skewness of spectra',
        units='',
    ),
    'correlation_coefficient': MetaData(
        long_name='Correlation coefficient',
    ),
    'spectral_differential_phase': MetaData(
        long_name='Spectral differential phase'
    ),
    'wind_direction': MetaData(
        long_name='Wind direction',
        units='degrees',
    ),
    'wind_speed': MetaData(
        long_name='Wind speed',
        units='m s-1',
    ),
    'Zdr': MetaData(
        long_name='Differential reflectivity',
        units='dB'
    ),
    'Ze': MetaData(
        long_name='Radar reflectivity factor.',
        units='dBZ',
    )
}
