"""Module for reading and processing Vaisala ceilometers."""
import linecache
import numpy as np


class VaisalaCeilo:
    """Base class for Vaisala ceilometers."""
    def __init__(self, file_name):
        self.file_name = file_name
        self._params = None
        self.backscatter = None
        self.metadata = None
        self.range = None

    def _calc_range(self):
        n_gates = int(self.metadata['number_of_gates'][0])
        range_resolution = int(self.metadata['range_resolution'][0])
        return np.arange(1, n_gates + 1) * range_resolution

    def _read_lines_in_file(self):
        """Returns file contents as list, one line per cell."""
        with open(self.file_name) as file:
            return file.readlines()

    @staticmethod
    def _get_empty_lines(data):
        """Returns indices of empty cells in list."""
        return [n for n, _ in enumerate(data) if isempty(data[n])]

    def _convert_data(self, lines):
        n_chars = self._params[0]
        n_gates = int((len(lines[0])-1)/n_chars)
        profiles = np.zeros((len(lines), n_gates), dtype=int)
        ran = range(0, n_gates*n_chars, n_chars)
        for ind, line in enumerate(lines):
            try:
                profiles[ind, :] = [int(line[i:i+n_chars], 16) for i in ran]
            except ValueError as error:
                print(error)
                profiles[ind, :] = np.zeros(n_gates, dtype=int)
        ind = np.where(profiles & self._params[1] != 0)
        profiles[ind] -= self._params[2]
        return profiles


class ClCeilo(VaisalaCeilo):
    """Base class for Vaisala CL31/CL51 ceilometers."""
    def __init__(self, file_name):
        super().__init__(file_name)
        self._params = (5, 524288, 1048576)

    def read_data(self):
        """Read all lines of data from the file."""
        all_lines = self._read_lines_in_file()
        empty_lines = self._get_empty_lines(all_lines)
        data_lines = self._parse_lines(all_lines, empty_lines)
        self.metadata = self._read_metadata(data_lines)
        self.backscatter = self._convert_data(data_lines[-1])
        self.range = self._calc_range()

    @classmethod
    def _read_metadata(cls, data_lines):
        header_line1 = cls._read_header_line_1(data_lines[1])
        header_line2 = cls._read_header_line_2(data_lines[2])
        header_line4 = cls._read_header_line_4(data_lines[-2])
        return {**header_line1, **header_line2, **header_line4}

    @staticmethod
    def _parse_lines(data, empty_lines):
        number_of_data_lines = empty_lines[1] - empty_lines[0] - 1
        lines = [[data[n+1][1:-1] for n in empty_lines],  # time stamp
                 [data[n+2][1:-2] for n in empty_lines]]
        for line_number in range(3, number_of_data_lines):
            lines.append([data[n+line_number] for n in empty_lines])
        return lines

    @staticmethod
    def _read_header_line_1(lines):
        keys = ('model_id', 'unit_id', 'software_version', 'message_number',
                'message_subclass')
        values = []
        for line in lines:
            values.append([line[:2], line[2:3], line[3:6], line[6], line[7]])
        return _values_to_dict(keys, values)

    @staticmethod
    def _read_header_line_2(lines):
        """Reading of Vaisala cloud base heights is not yet included here."""
        keys = ('detection_status', 'warning')
        values = []
        for line in lines:
            values.append([line[0], line[1], line[2:]])
        return _values_to_dict(keys, values)

    @staticmethod
    def _read_header_line_4(lines):
        keys = ('scale', 'range_resolution', 'number_of_gates', 'laser_energy',
                'laser_temperature', 'window_transmission',
                'receiver_sensitivity', 'window_contamination')
        values = []
        for line in lines:
            values.append(line.split())
        return _values_to_dict(keys, values)


def _values_to_dict(keys, values):
    out = {}
    for i, key in enumerate(keys):
        out[key] = np.array([x[i] for x in values])
    return out


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
    """Class for Vaisala CT25k ceilometer."""
    def __init__(self, input_file):
        super().__init__(input_file)
        self.model = 'ct25k'

    def read_data(self):
        """CT25k need special methods for reading."""
        pass


def ceilo2nc(input_file, output_file):
    """Converts Vaisala ceilometer txt-file to netCDF."""
    ceilo = _initialize_ceilo(input_file)
    ceilo.read_data()


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
            if isempty(line):
                break
            line_number += 1
    return line_number


def isempty(line):
    """Tests if line in text file is empty."""
    if line in ('\n', '\r\n'):
        return True
    return False
