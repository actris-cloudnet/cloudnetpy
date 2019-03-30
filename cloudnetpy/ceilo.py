import linecache
import numpy as np


class VaisalaCeilo:
    """Base class for Vaisala ceilometers."""
    def __init__(self, file_name):
        self.file_name = file_name
        self.params = None

    def _read_lines_in_file(self):
        """Returns file contents as list, one line per cell."""
        with open(self.file_name) as f:
            return f.readlines()

    @staticmethod
    def _get_empty_lines(data):
        """Returns indices of empty cells in list."""
        return [n for n, _ in enumerate(data) if isempty(data[n])]

    def _convert_data(self, lines, n_gates):
        assert min(n_gates) == max(n_gates), 'Error: variable number of range gates in profile.'
        n_chars = self.params[0]
        n_gates = n_gates[0]
        ran = range(0, n_gates*n_chars, n_chars)
        profiles = np.zeros((len(lines), n_gates), dtype=int)
        for ind, line in enumerate(lines):
            try:
                profiles[ind, :] = [int(line[i:i+n_chars], 16) for i in ran]
            except ValueError as error:
                profiles[ind, :] = np.zeros(n_gates, dtype=int)
        ind = np.where(profiles & self.params[1] != 0)
        profiles[ind] -= self.params[2]
        return profiles


class ClCeilo(VaisalaCeilo):
    """Base class for more modern Vaisala CL31/CL51 ceilometers."""
    def __init__(self, file_name):
        super().__init__(file_name)
        self.params = (5, 524288, 1048576)

    def read_data(self):
        """Read all lines of data from the file."""
        data = self._read_lines_in_file()
        empty_lines = self._get_empty_lines(data)
        lines = self._parse_lines(data, empty_lines)
        header_line1 = self._read_header_line_1(lines[1])
        header_line4 = self._read_header_line_4(lines[4])
        profiles = self._convert_data(lines[5], header_line4['number_of_gates'])

    @staticmethod
    def _parse_lines(data, empty_lines):
        lines = [[data[n+1][1:-1] for n in empty_lines],
                 [data[n+2][1:-2] for n in empty_lines]]
        for m in range(3, 7):
            lines.append([data[n+m] for n in empty_lines])
        return lines

    @staticmethod
    def _read_header_line_1(lines):
        keys = ('model_id', 'unit_id', 'software_version', 'message_number',
                'message_subclass')
        values, out = [], {}
        for line in lines:
            values.append([line[:2], line[2:3], line[3:6], line[6], line[7]])
        for i, key in enumerate(keys):
            out[key] = np.array([x[i] for x in values])
        return out

    @staticmethod
    def _read_header_line_4(lines):
        keys = ('range_resolution', 'number_of_gates', 'laser_energy',
                'laser_temperature', 'window_transmission',
                'receiver_sensitivity', 'window_contamination')
        values, out = [], {}
        for line in lines:
            values.append(line.split())
        for i, key in enumerate(keys):
            out[key] = np.array([int(x[i+1]) for x in values])
        return out


class Cl51(ClCeilo):
    def __init__(self, input_file):
        super().__init__(input_file)
        self.model = 'cl51'


class Cl31(ClCeilo):
    def __init__(self, input_file):
        super().__init__(input_file)
        self.model = 'cl31'


class Ct25k(VaisalaCeilo):
    def __init__(self, input_file):
        super().__init__(input_file)
        self.model = 'ct25k'

    def read_data(self):
        """CT25k need special methods for reading."""
        pass


def ceilo2nc(input_file, output_file):
    cl = _initialize_ceilo(input_file)
    cl.read_data()


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


def _find_first_empty_line(file):
    line_number = 0
    while True:
        line = linecache.getline(file, line_number)
        if isempty(line):
            return line_number
        line_number += 1


def isempty(line):
    if line in ('\n', '\r\n'):
        return True
    return False
