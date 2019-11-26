"""Module for reading RPG 94 GHz radar header."""
import numpy as np


def read_rpg_header(file_name, level, version=3):
    """Reads header from RPG binary file.

    Supports LV0 / LV1 files and version 2 / 3.

    Args:
        file_name (str): name of the file.
        level (int): File level (0 or 1).
        version (int, optional): RPG software version (2 or 3). Default is 3.

    Returns:
        tuple: 2-element tuple containing the header (as dict) and file position.

    """
    def _read_block(*fields):
        block = np.fromfile(file, np.dtype(list(fields)), 1)
        for name in block.dtype.names:
            array = block[name][0]
            header[name] = np.array(array, dtype=_get_dtype(array))

    header = {}
    file = open(file_name, 'rb')

    _read_block(('file_code', 'i4'),
                ('header_length', 'i4'))  # bytes

    if version >= 3:
        _read_block(('start_time', 'uint32'),
                    ('stop_time', 'uint32'))

    _read_block(('program_number', 'i4'),
                ('model_number', 'i4'))  # 0 = Single pol. 1 = Dual pol.

    header['program_name'] = _read_string(file)
    header['customer_name'] = _read_string(file)

    _read_block(('radar_frequency', 'f'),  # GHz
                ('antenna_separation', 'f'),  # m
                ('antenna_diameter', 'f'),  # m
                ('antenna_gain', 'f'),
                ('half_power_beam_width', 'f'))  # degrees

    if level == 0:
        _read_block(('radar_constant', 'f'))

    _read_block(('dual_polarization', 'i1'))  # 0=Single, 1=Dual (LDR), 2=Dual (STSR)

    if level == 0:
        _read_block(('compression', 'i1'),  # 0=Not compressed, 1=Compressed
                    ('anti_alias', 'i1'))  # 0=Not anti-aliased, 1=Anti-aliased

    _read_block(('sample_duration', 'f'),  # s
                ('latitude', 'f'),
                ('longitude', 'f'),
                ('calibration_interval', 'i4'),
                ('number_of_range_levels', 'i4'),
                ('number_of_temperature_levels', 'i4'),
                ('number_of_humidity_levels', 'i4'),
                ('number_of_chirp_levels', 'i4'))

    n_levels, n_temp, n_humidity, n_chirp = _get_number_of_levels(header)

    _read_block(('range', _dim(n_levels)),
                ('temperature_levels', _dim(n_temp)),
                ('humidity_levels', _dim(n_humidity)))

    if level == 0:
        _read_block(('range_factors', _dim(n_levels)))

    _read_block(('n_spectral_samples', _dim(n_chirp, 'i4')),
                ('chirp_start_indices', _dim(n_chirp, 'i4')),
                ('n_averaged_chirps', _dim(n_chirp, 'i4')),
                ('integration_time', _dim(n_chirp)),  # s
                ('range_resolution', _dim(n_chirp)),  # m
                ('nyquist_velocity', _dim(n_chirp)))  # m/s

    if version > 2:
        if level == 0:
            _read_block(('channel_bandwidth', _dim(n_chirp)),  # Hz
                        ('chirp_low_if', _dim(n_chirp, 'i4')),  # Hz
                        ('chirp_high_if', _dim(n_chirp, 'i4')),  # Hz
                        ('range_min', _dim(n_chirp, 'i4')),  # m
                        ('range_max', _dim(n_chirp, 'i4')),  # m
                        ('chirp_fft_size', _dim(n_chirp, 'i4')),  # Must be power of 2
                        ('n_invalid_samples', _dim(n_chirp, 'i4')),
                        ('chirp_center_freq', _dim(n_chirp)),  # MHz
                        ('chirp_bandwidth', _dim(n_chirp)),  # MHz
                        ('fft_start_ind', _dim(n_chirp, 'i4')),
                        ('fft_stop_ind', _dim(n_chirp, 'i4')),
                        ('chrp_fft_no', _dim(n_chirp, 'i4')),
                        ('adc_sample_rate', 'i4'),  # Hz
                        ('max_range', 'i4'))  # m

        _read_block(('is_power_levelling', 'i1'),  # 0=no, 1=yes
                    ('is_spike_filter', 'i1'),  # 0=no, 1=yes
                    ('is_phase_correction', 'i1'),  # 0=no, 1=yes
                    ('is_relative_power_correction', 'i1'),  # 0=no, 1=yes
                    ('fft_window', 'i1'),  # 0=square, 1=parzen, 2=blackman, 3=welch, 4=slepian2, 5=slepian3
                    ('adc_input_voltage_range', 'i4'),  # mV
                    ('noise_filter_threshold', 'f4'))  # multiple of STD in spectra

        if level == 0:
            _read_block(('dummy1', _dim(25, 'i4')),
                        ('dummy2', _dim(10000, 'uint32')))

    file_position = file.tell()
    file.close()

    return header, file_position


def _read_string(file_id):
    """Read characters from binary data until whitespace."""
    str_out = ''
    while True:
        c = np.fromfile(file_id, np.int8, 1)
        if c:
            if c < 0:
                c = 0
            str_out += chr(c)
        else:
            break
    return str_out


def _get_number_of_levels(header):
    for name in ('range', 'temperature', 'humidity', 'chirp'):
        yield int(header[f"number_of_{name}_levels"])


def _dim(length, dtype='f'):
    return f"({length},){dtype}"


def _get_dtype(array):
    if array.dtype in (np.int8, np.int32, np.uint32):
        return int
    return float
