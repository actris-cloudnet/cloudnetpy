import logging
from collections import namedtuple
import numpy as np
from cloudnetpy import utils


class Fmcw94Bin:
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
            if len(c) == 1 and c[0] < 0:
                c = [63]
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
            vrs = {'_sample_length': np.zeros(dims.n_samples, int),
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
                'azimuth_angle',
                'status_flag',
                'transmitted_power',
                'transmitter_temperature',
                'receiver_temperature',
                'pc_temperature'))

            block2_vars = dict.fromkeys((
                'Zh',
                'v',
                'width',
                'skewness',
                'kurtosis'))

            if int(self.header['dual_polarization'][0]) == 1:
                block2_vars.update(dict.fromkeys((
                    'ldr',
                    'rho_cx',
                    'phi_cx')))
            elif int(self.header['dual_polarization'][0]) == 2:
                block2_vars.update(dict.fromkeys((
                    'zdr',
                    'rho_hv',
                    'phi_dp',
                    '_',
                    'sldr',
                    'srho_hv',
                    'kdp',
                    'differential_attenuation')))
            return vrs, block1_vars, block2_vars

        def _add_sensitivities():
            ind0 = len(block1) + n_dummy
            ind1 = ind0 + dims.n_gates
            block1['_sensitivity_limit_v'] = float_block1[:, ind0:ind1]
            if int(self.header['dual_polarization'][0]) > 0:
                block1['_sensitivity_limit_h'] = float_block1[:, ind1:]

        def _get_length_of_dummy_data():
            return 3 + dims.n_layers_t + 2*dims.n_layers_h

        def _get_length_of_sensitivity_data():
            if int(self.header['dual_polarization'][0]) > 0:
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
            aux['_sample_length'][sample] = np.fromfile(file, np.int32, 1)
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


class HatproBin:
    """HATPRO binary file reader."""
    def __init__(self, filename):
        self.filename = filename
        self._file_position = 0
        self.header = self.read_header()
        self.data = self.read_data()

    def screen_bad_profiles(self):
        good_ind = []
        for ind, flag in enumerate(self.data['quality_flag']):
            if not (utils.isbit(flag, 1) and utils.isbit(flag, 2)):
                good_ind.append(ind)
        for key in self.data.keys():
            self.data[key] = self.data[key][good_ind]

    def read_header(self) -> dict:
        """Reads the header."""
        file = open(self.filename, 'rb')
        header = {
            'file_code': np.fromfile(file, np.int32, 1),
            '_n_samples': np.fromfile(file, np.int32, 1),
            '_lwp_min_max': np.fromfile(file, np.float32, 2),
            '_time_reference': np.fromfile(file, np.int32, 1),
            'retrieval_method': np.fromfile(file, np.int32, 1)
        }
        self._file_position = file.tell()
        file.close()
        return header

    def read_data(self) -> dict:
        """Reads the data."""
        file = open(self.filename, 'rb')
        file.seek(self._file_position)

        data = {
            'time': np.zeros(self.header['_n_samples'], dtype=np.int32),
            'quality_flag': np.zeros(self.header['_n_samples'], dtype=np.int32),
            'lwp': np.zeros(self.header['_n_samples']),
            'zenith_angle': np.zeros(self.header['_n_samples'], dtype=np.float32)
        }

        version = self._get_hatpro_version()
        angle_dtype = np.float32 if version == 1 else np.int32
        data['_instrument_angles'] = np.zeros(self.header['_n_samples'], dtype=angle_dtype)

        for sample in range(self.header['_n_samples'][0]):
            data['time'][sample] = np.fromfile(file, np.int32, 1)
            data['quality_flag'][sample] = np.fromfile(file, np.int8, 1)
            data['lwp'][sample] = np.fromfile(file, np.float32, 1)
            data['_instrument_angles'][sample] = np.fromfile(file, angle_dtype, 1)

        data = self._add_zenith(version, data)

        file.close()
        return data

    def _get_hatpro_version(self) -> int:
        if self.header['file_code'][0] == 934501978:
            return 1
        if self.header['file_code'][0] == 934501000:
            return 2
        raise ValueError(f'Unknown HATPRO version. {self.header["file_code"][0]}')

    @staticmethod
    def _add_zenith(version: int, data: dict) -> dict:
        if version == 1:
            del data['zenith_angle']  # Impossible to understand how zenith is decoded in the values
        else:
            elevation_angle = [int(str(x)[:4])/100 for x in data['_instrument_angles']]
            data['zenith_angle'] = 90 - np.array(elevation_angle)
        return data
