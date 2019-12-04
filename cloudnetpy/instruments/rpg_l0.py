from cloudnetpy.instruments.rpg_header import read_rpg_header
from collections import namedtuple
import numpy as np
import sys
import bisect


class RpgBinL0:
    """RPG Cloud Radar Level 0 v3 data reader."""
    def __init__(self, filename, level):
        self.filename = filename
        self.level = level
        self.header, self._file_position = read_rpg_header(filename, level)
        self.data = self.read_rpg_data()

    def read_rpg_data(self):
        """Reads the actual data from rpg binary file."""

        def _create_dimensions():
            """Returns possible lengths of the data arrays."""
            Dimensions = namedtuple('Dimensions', ['n_samples',
                                                   'n_gates',
                                                   'n_layers_t',
                                                   'n_layers_h'])
            return Dimensions(int(np.fromfile(file, np.int32, 1)),
                              int(self.header['n_range_levels']),
                              int(self.header['n_temperature_levels']),
                              int(self.header['n_humidity_levels']))

        def _create_variables():
            """Initializes dictionaries for data arrays."""
            vrs = {'sample_length': np.zeros(dims.n_samples, np.int),
                   'time': np.zeros(dims.n_samples, np.int),
                   'time_ms': np.zeros(dims.n_samples, np.int),
                   'quality_flag': np.zeros(dims.n_samples, np.int)}

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

            if self.level == 1:

                block2_vars = dict.fromkeys((
                    'Ze',
                    'v',
                    'width',
                    'skewness',
                    'kurtosis'))

                if self.header['dual_polarization'] > 0:
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

            else:

                block2_vars = {}

                if self.header['compression'] == 0:

                    block2_vars['doppler_spectrum'] = None

                    if self.header['dual_polarization'] > 0:
                        block2_vars.update(dict.fromkeys((
                            'doppler_spectrum_h',
                            'covariance_spectrum_re',
                            'covariance_spectrum_im')))

                elif self.header['compression'] > 0:

                    block2_vars.update(dict.fromkeys(
                        'doppler_spectrum_compressed'))

                    if self.header['dual_polarization'] > 0:
                        block2_vars.update(dict.fromkeys((
                            'doppler_spectrum_h_compressed',
                            'covariance_spectrum_re_compressed',
                            'covariance_spectrum_im_compressed')))

                if self.header['compression'] == 2:

                    block2_vars.update(dict.fromkeys((
                        'differential_reflectivity_compressed',
                        'spectral_correlation_coefficient_compressed',
                        'spectral_differential_phase_compressed')))

                    if self.header['dual_polarization'] == 2:
                        block2_vars.update(dict.fromkeys((
                            'spectral_slanted_ldr_compressed',
                            'spectral_slanted_correlation_coefficient_compressed')))

            return vrs, block1_vars, block2_vars

        def _get_float_block_lengths():
            block_one_length = len(block1) + 3 + dims.n_layers_t + (2*dims.n_layers_h) + (2*dims.n_gates)
            if self.level == 0 and self.header['dual_polarization'] > 0:
                block_one_length += 2*dims.n_gates
            block_two_length = len(block2)
            return block_one_length, block_two_length

        def _init_float_blocks():
            block_one = np.zeros((dims.n_samples, n_floats1))
            if self.level == 0:
                block_two = np.zeros((dims.n_samples, dims.n_gates, max(n_spectral_samples)))
            else:
                block_two = np.zeros((dims.n_samples, dims.n_gates, n_floats2))
            return block_one, block_two

        file = open(self.filename, 'rb')
        file.seek(self._file_position)
        dims = _create_dimensions()
        aux, block1, block2 = _create_variables()
        n_floats1, n_floats2 = _get_float_block_lengths()
        n_spectral_samples = self.header['n_spectral_samples']
        chirp_indices = self.header['chirp_start_indices']
        float_block1, float_block2 = _init_float_blocks()

        for sample in range(dims.n_samples):

            aux['sample_length'][sample] = np.fromfile(file, np.int32, 1)
            aux['time'][sample] = np.fromfile(file, np.uint32, 1)
            aux['time_ms'][sample] = np.fromfile(file, np.int32, 1)
            aux['quality_flag'][sample] = np.fromfile(file, np.int8, 1)
            float_block1[sample, :] = np.fromfile(file, np.float32, n_floats1)
            is_data_ind = np.where(np.fromfile(file, np.int8, dims.n_gates))[0]

            if self.level == 1:

                n_valid = len(is_data_ind)
                values = np.fromfile(file, np.float32, n_floats2 * n_valid)
                float_block2[sample, is_data_ind, :] = values.reshape(n_valid, n_floats2)

            elif self.header['compression'] == 0:

                n_var = 4 if self.header['dual_polarization'] > 0 else 1
                n_samples = [n_spectral_samples[bisect.bisect(chirp_indices, x)-1]
                             for x in is_data_ind]
                dtype = ' '.join([f"int32, ({n_var*x},)float32, " for x in n_samples])
                data_array = np.array(np.fromfile(file, np.dtype(dtype), 1)
                                      [0].tolist())[1::2]
                for n, ind in enumerate(is_data_ind):
                    float_block2[sample, ind, :n_samples[n]] = data_array[n][:n_samples[n]]

            elif self.header['compression'] > 0:

                for _ in is_data_ind:

                    n_bytes_in_block = np.fromfile(file, np.int32, 1)
                    n_blocks = int(np.fromfile(file, np.int8, 1)[0])
                    min_ind, max_ind = np.fromfile(file, np.dtype(f"({n_blocks}, )int16"), 2)
                    n_indices = max_ind - min_ind

                    n_values = (sum(n_indices) + len(n_indices)) * 4 + 2
                    all_data = np.fromfile(file, np.float32, n_values)

                    if self.header['anti_alias'] == 1:
                        is_anti_applied, min_velocity = np.fromfile(file, np.dtype('int8, float32'), 1)[0]

        file.close()
        for n, name in enumerate(block1):
            block1[name] = float_block1[:, n]  # with l0 there is still stuff in end of block1 after this

        #if self.level == 1:
        #    for n, name in enumerate(block2):
        #        block2[name] = float_block2[:, :, n]
        #else:


        return {**aux, **block1, **block2}


