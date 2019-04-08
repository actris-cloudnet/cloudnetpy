"""Module for reading and processing Vaisala ceilometers."""
import linecache
import numpy as np
from cloudnetpy import plotting
import matplotlib.pyplot as plt
import sys


class VaisalaCeilo:
    """Base class for Vaisala ceilometers."""
    def __init__(self, file_name):
        self.file_name = file_name
        self.model = None
        self.message_number = None
        self.hex_conversion_params = None
        self.backscatter = None
        self.metadata = None
        self.range = None
        self.time = None

    def _fetch_data_lines(self):
        """Finds data lines (header + backscatter) from ceilometer file."""
        with open(self.file_name) as file:
            all_lines = file.readlines()
        return self._screen_empty_lines(all_lines)
        
    @staticmethod
    def _screen_empty_lines(data):
        """Removes empty lines from the list of data."""

        def _parse_empty_lines():
            return [n for n, _ in enumerate(data) if is_empty_line(data[n])]

        def _parse_data_lines(empty_indices):
            number_of_data_lines = empty_indices[1] - empty_indices[0] - 1
            lines = []
            for line_number in range(number_of_data_lines):
                lines.append([data[n + line_number + 1] for n in empty_indices])
            return lines

        empty_lines = _parse_empty_lines()
        data_lines = _parse_data_lines(empty_lines)
        return data_lines

    def _read_header_line_1(self, lines):
        """Reads all first header lines from CT25k and CL ceilometers."""
        keys = ('model_id', 'unit_id', 'software_version', 'message_number',
                'message_subclass')
        values = []
        if 'cl' in self.model:
            indices = [1, 3, 4, 7, 8, 9]
        else:
            indices = [1, 3, 4, 6, 7, 8]
        for line in lines:
            distinct_values = _split_string(line, indices)
            values.append(distinct_values)
        return _values_to_dict(keys, values)

    @staticmethod
    def _read_header_line_2(lines):
        """Same for all data messages."""
        keys = ('detection_status', 'warning', 'cloud_base_data', 'warning_flags')
        values = []
        for line in lines:
            distinct_values = [line[0], line[1], line[3:20], line[21:].strip()]
            values.append(distinct_values)
        return _values_to_dict(keys, values)

    @staticmethod
    def _get_message_number(header_line_1):
        msg_no = header_line_1['message_number']
        assert len(np.unique(msg_no)) == 1, 'Error: inconsistent message numbers.'
        return int(msg_no[0])

    def _read_backscatter(self, lines):
        n_chars = self.hex_conversion_params[0]
        n_gates = int(len(lines[0])/n_chars)
        profiles = np.zeros((len(lines), n_gates), dtype=int)
        ran = range(0, n_gates*n_chars, n_chars)
        for ind, line in enumerate(lines):
            try:
                profiles[ind, :] = [int(line[i:i+n_chars], 16) for i in ran]
            except ValueError as error:
                print(error)

        ind = np.where(profiles & self.hex_conversion_params[1] != 0)
        profiles[ind] -= self.hex_conversion_params[2]
        return profiles

    @staticmethod
    def _calc_time(time_lines):
        time = [time_to_fraction_hour(line.split()[1]) for line in time_lines]
        return np.array(time)

    def _calc_range(self):
        n_gates = int(self.metadata['number_of_gates'][0])
        range_resolution = int(self.metadata['range_resolution'][0])
        return np.arange(1, n_gates + 1) * range_resolution

    
class ClCeilo(VaisalaCeilo):
    """Base class for Vaisala CL31/CL51 ceilometers."""

    def __init__(self, file_name):
        super().__init__(file_name)
        self.hex_conversion_params = (5, 524288, 1048576)

    def read_ceilometer_file(self):
        """Read all lines of data from the file."""
        data_lines = self._fetch_data_lines()
        header_line_1 = self._read_header_line_1(data_lines[1])
        self.message_number = self._get_message_number(header_line_1)
        header_line_2 = self._read_header_line_2(data_lines[2])
        header_line_3 = self._read_header_line_3(data_lines[3])
        header_line_4 = self._read_header_line_4(data_lines[-3])
        self.backscatter = self._read_backscatter(data_lines[-2])

    def _read_header_line_3(self, lines):
        if self.message_number != 2:
            return None
        keys = ('cloud_detection_status', 'cloud_amount_data')
        values = []
        for line in lines:
            distinct_values = [line[0:3], line[3:].strip()]
            values.append(distinct_values)
        return _values_to_dict(keys, values)

    @staticmethod
    def _read_header_line_4(lines):
        keys = ('scale', 'range_resolution', 'number_of_gates', 'laser_energy',
                'laser_temperature', 'window_transmission', 'tilt_angle',
                'background_light', 'measurement_parameters', 'backscatter_sum')
        values = []
        for line in lines:
            values.append(line.split())
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
        self.hex_conversion_params = (4, 32768, 65536)

    def read_ceilometer_file(self):
        """Read all lines of data from the file."""
        data_lines = self._fetch_data_lines()
        header_line_1 = self._read_header_line_1(data_lines[1])
        self.message_number = self._get_message_number(header_line_1)
        header_line_2 = self._read_header_line_2(data_lines[2])
        header_line_3 = self._read_header_line_3(data_lines[3])
        hex_profiles = self._parse_hex_profiles(data_lines[4:20])
        self.backscatter = self._read_backscatter(hex_profiles)

    @staticmethod
    def _parse_hex_profiles(lines):
        """Collects ct25k profiles into list (one profile / element)."""
        n_profiles = len(lines[0])
        return [''.join([lines[l][n][3:].strip() for l in range(16)])
                for n in range(n_profiles)]

    def _read_header_line_3(self, lines):
        if self.message_number in (1, 3, 6):
            return None
        keys = ('scale', 'measurement_mode', 'laser_energy',
                'laser_temperature', 'receiver_sensitivity',
                'window_contamination', 'tilt_angle', 'background_light',
                'measurement_parameters', 'backscatter_sum')
        values = []
        for line in lines:
            values.append(line.split())
        return _values_to_dict(keys, values)


def ceilo2nc(input_file, output_file):
    """Converts Vaisala ceilometer txt-file to netCDF."""
    ceilo = _initialize_ceilo(input_file)
    ceilo.read_ceilometer_file()


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
    return Ct25k(file)


def _find_ceilo_model(file):
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
    return int(h) + (int(m) * 60 + int(s)) / 3600
