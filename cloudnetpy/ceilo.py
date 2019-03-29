import linecache
import sys
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils


class VaisalaCeilo:
    """Base class for Vaisala ceilometers."""
    def __init__(self, file_name):
        self.file_name = file_name
        self.data = None

    def _read_lines_in_file(self):
        """Returns file contents as list, one line per cell."""
        with open(self.file_name) as f:
            return f.readlines()

    @staticmethod
    def _get_empty_lines(data):
        """Returns indices of empty cells in list."""
        return [n for n, _ in enumerate(data) if isempty(data[n])]


class ClCeilo(VaisalaCeilo):
    """Base class for more modern Vaisala CL31/CL51 ceilometers."""
    def __init__(self, file_name):
        super().__init__(file_name)

    def read_data(self):
        """Read all lines of data from the file."""
        data = self._read_lines_in_file()
        empty_lines = self._get_empty_lines(data)
        lines = [[data[n+1][1:-1] for n in empty_lines],
                 [data[n+2][1:-2] for n in empty_lines]]
        for m in range(3, 7):
            lines.append([data[n+m] for n in empty_lines])
        self.data = lines


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
    index = 0
    while True:
        line = linecache.getline(file, index)
        if line in ('\n', '\r\n'):
            return index
        index += 1


def isempty(line):
    if line in ('\n', '\r\n'):
        return True
    return False
