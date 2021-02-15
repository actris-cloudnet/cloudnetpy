"""Module with classes for Vaisala ceilometers."""
from typing import Tuple
import numpy as np
from cloudnetpy.instruments.ceilometer import Ceilometer
from cloudnetpy import utils


M2KM = 0.001
SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 3600


class VaisalaCeilo(Ceilometer):
    """Base class for Vaisala ceilometers."""
    def __init__(self, full_path: str):
        super().__init__(full_path)
        self._backscatter_scale_factor = 1
        self._hex_conversion_params = (1, 1, 1)
        self._message_number = None

    def _fetch_data_lines(self) -> list:
        """Finds data lines (header + backscatter) from ceilometer file."""
        with open(self.file_name) as file:
            all_lines = file.readlines()
        return self._screen_invalid_lines(all_lines)

    def _calc_range(self) -> np.ndarray:
        """Calculates range vector from the resolution and number of gates."""
        if self.model == 'ct25k':
            range_resolution = 30
            n_gates = 256
        else:
            n_gates = int(self.metadata['number_of_gates'])
            range_resolution = int(self.metadata['range_resolution'])
        return np.arange(n_gates)*range_resolution + range_resolution/2

    def _read_backscatter(self, lines: list) -> np.ndarray:
        """Converts backscatter profile from 2-complement hex to floats."""
        n_chars = self._hex_conversion_params[0]
        n_gates = int(len(lines[0])/n_chars)
        profiles = np.zeros((len(lines), n_gates), dtype=int)
        ran = range(0, n_gates*n_chars, n_chars)
        for ind, line in enumerate(lines):
            try:
                profiles[ind, :] = [int(line[i:i+n_chars], 16) for i in ran]
            except ValueError:
                print('Warning: bad value in raw ceilometer data')
        ind = np.where(profiles & self._hex_conversion_params[1] != 0)
        profiles[ind] -= self._hex_conversion_params[2]
        return profiles.astype(float) / self._backscatter_scale_factor

    @staticmethod
    def _screen_invalid_lines(data: list) -> list:
        """Removes empty (and other weird) lines from the list of data."""

        def _find_timestamp_line_numbers() -> list:
            return [n for n, _ in enumerate(data) if utils.is_timestamp(data[n])]

        def _find_number_of_data_lines(timestamp_line_number: int) -> int:
            for i, line in enumerate(data[timestamp_line_number:]):
                if utils.is_empty_line(line):
                    return i

        def _parse_data_lines(starting_indices: list) -> list:
            return [[data[n + line_number] for n in starting_indices]
                    for line_number in range(number_of_data_lines)]

        timestamp_line_numbers = _find_timestamp_line_numbers()
        number_of_data_lines = _find_number_of_data_lines(timestamp_line_numbers[0])
        return _parse_data_lines(timestamp_line_numbers)

    @staticmethod
    def _get_message_number(header_line_1: dict) -> int:
        msg_no = header_line_1['message_number']
        assert len(np.unique(msg_no)) == 1, 'Error: inconsistent message numbers.'
        return int(msg_no[0])

    @staticmethod
    def _calc_time(time_lines: list) -> np.ndarray:
        """Returns the time vector as fraction hour."""
        time = [time_to_fraction_hour(line.split()[1]) for line in time_lines]
        return np.array(time)

    @staticmethod
    def _calc_date(time_lines) -> list:
        """Returns the date [yyyy, mm, dd]"""
        return time_lines[0].split()[0].strip('-').split('-')

    @classmethod
    def _handle_metadata(cls, header: list) -> dict:
        meta = cls._concatenate_meta(header)
        meta = cls._remove_meta_duplicates(meta)
        meta = cls._convert_meta_strings(meta)
        return meta

    @staticmethod
    def _concatenate_meta(header: list) -> dict:
        meta = {}
        for head in header:
            meta.update(head)
        return meta

    @staticmethod
    def _remove_meta_duplicates(meta: dict) -> dict:
        for field in meta:
            if len(np.unique(meta[field])) == 1:
                meta[field] = meta[field][0]
        return meta

    @staticmethod
    def _convert_meta_strings(meta: dict) -> dict:
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

    def _read_common_header_part(self) -> Tuple[list, list]:
        header = []
        data_lines = self._fetch_data_lines()
        self.time = self._calc_time(data_lines[0])
        self.date = self._calc_date(data_lines[0])
        header.append(self._read_header_line_1(data_lines[1]))
        self._message_number = self._get_message_number(header[0])
        header.append(self._read_header_line_2(data_lines[2]))
        return header, data_lines

    def _read_header_line_1(self, lines: list) -> dict:
        """Reads all first header lines from CT25k and CL ceilometers."""
        fields = ('model_id', 'unit_id', 'software_level', 'message_number', 'message_subclass')
        if 'cl' in self.model:
            indices = [1, 3, 4, 7, 8, 9]
        else:
            indices = [1, 3, 4, 6, 7, 8]
        values = [split_string(line, indices) for line in lines]
        return values_to_dict(fields, values)

    @staticmethod
    def _read_header_line_2(lines: list) -> dict:
        """Reads the second header line."""
        fields = ('detection_status', 'warning', 'cloud_base_data',
                  'warning_flags')
        values = [[line[0], line[1], line[3:20], line[21:].strip()] for line in lines]
        return values_to_dict(fields, values)

    def _range_correct_upper_part(self) -> None:
        altitude_limit = 2400
        ind = np.where(self.range > altitude_limit)
        self.backscatter[:, ind] *= (self.range[ind]*M2KM)**2


class ClCeilo(VaisalaCeilo):
    """Base class for Vaisala CL31/CL51 ceilometers."""

    def __init__(self, full_path: str):
        super().__init__(full_path)
        self._hex_conversion_params = (5, 524288, 1048576)
        self._backscatter_scale_factor = 1e8
        self.noise_params = (100, 1e-12, 3e-6, (1.1e-8, 2.9e-8))

    def read_ceilometer_file(self) -> None:
        """Read all lines of data from the file."""
        header, data_lines = self._read_common_header_part()
        header.append(self._read_header_line_4(data_lines[-3]))
        self.metadata = self._handle_metadata(header)
        self.range = self._calc_range()
        self.backscatter = self._read_backscatter(data_lines[-2])

    def _read_header_line_3(self, lines: list) -> dict:
        if self._message_number != 2:
            raise RuntimeError('Unsupported message number.')
        keys = ('cloud_detection_status', 'cloud_amount_data')
        values = [[line[0:3], line[3:].strip()] for line in lines]
        return values_to_dict(keys, values)

    @staticmethod
    def _read_header_line_4(lines: list) -> dict:
        keys = ('scale', 'range_resolution', 'number_of_gates', 'laser_energy',
                'laser_temperature', 'window_transmission', 'tilt_angle',
                'background_light', 'measurement_parameters', 'backscatter_sum')
        values = [line.split() for line in lines]
        return values_to_dict(keys, values)


class Cl51(ClCeilo):
    """Class for Vaisala CL51 ceilometer."""
    def __init__(self, full_path: str):
        super().__init__(full_path)
        self.model = 'cl51'
        self.wavelength = 915


class Cl31(ClCeilo):
    """Class for Vaisala CL31 ceilometer."""
    def __init__(self, full_path: str):
        super().__init__(full_path)
        self.model = 'cl31'
        self.wavelength = 910


class Ct25k(VaisalaCeilo):
    """Class for Vaisala CT25k ceilometer.

    References:
        https://www.manualslib.com/manual/1414094/Vaisala-Ct25k.html

    """
    def __init__(self, input_file: str):
        super().__init__(input_file)
        self.model = 'ct25k'
        self._hex_conversion_params = (4, 32768, 65536)
        self._backscatter_scale_factor = 1e7
        self.noise_params = (40, 2e-14, 0.3e-6, (3e-10, 1.5e-9))
        self.wavelength = 905

    def read_ceilometer_file(self) -> None:
        """Read all lines of data from the file."""
        header, data_lines = self._read_common_header_part()
        header.append(self._read_header_line_3(data_lines[3]))
        self.metadata = self._handle_metadata(header)
        self.range = self._calc_range()
        hex_profiles = self._parse_hex_profiles(data_lines[4:20])
        self.backscatter = self._read_backscatter(hex_profiles)
        # TODO: should study the background noise to determine if the
        # next call is needed. It can be the case with cl31/51 also.
        self._range_correct_upper_part()

    @staticmethod
    def _parse_hex_profiles(lines: list) -> list:
        """Collects ct25k profiles into list (one profile / element)."""
        n_profiles = len(lines[0])
        return [''.join([lines[m][n][3:].strip() for m in range(16)]) for n in range(n_profiles)]

    def _read_header_line_3(self, lines: list) -> dict:
        if self._message_number in (1, 3, 6):
            raise RuntimeError('Unsupported message number.')
        keys = ('scale', 'measurement_mode', 'laser_energy',
                'laser_temperature', 'receiver_sensitivity',
                'window_contamination', 'tilt_angle', 'background_light',
                'measurement_parameters', 'backscatter_sum')
        values = [line.split() for line in lines]
        return values_to_dict(keys, values)


def split_string(string: str, indices: list) -> list:
    """Splits string between indices.

    Notes:
        It is possible to skip characters from the beginning and end of the
        string but not from the middle.

    Examples:
        >>> s = 'abcde'
        >>> indices = [1, 2, 4]
        >>> split_string(s, indices)
        ['b', 'cd']

    """
    return [string[n:m] for n, m in zip(indices[:-1], indices[1:])]


def values_to_dict(keys: tuple, values: list) -> dict:
    """Converts list elements to dictionary.

    Examples:
        >>> keys = ('a', 'b')
        >>> values = [[1, 2], [1, 2], [1, 2], [1, 2]]
        >>> values_to_dict(keys, values)
        {'a': array([1, 1, 1, 1]), 'b': array([2, 2, 2, 2])}

    """
    out = {}
    for i, key in enumerate(keys):
        out[key] = np.array([x[i] for x in values])
    return out


def time_to_fraction_hour(time: str) -> float:
    """Returns time (hh:mm:ss) as fraction hour """
    hour, minute, sec = time.split(':')
    return int(hour) + (int(minute) * SECONDS_IN_MINUTE + int(sec)) / SECONDS_IN_HOUR
