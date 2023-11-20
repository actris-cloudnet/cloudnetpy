"""Module with classes for Vaisala ceilometers."""
import itertools
import logging

import numpy as np

from cloudnetpy import utils
from cloudnetpy.constants import SEC_IN_HOUR, SEC_IN_MINUTE
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.ceilometer import Ceilometer, NoiseParam

M2KM = 0.001


class VaisalaCeilo(Ceilometer):
    """Base class for Vaisala ceilometers."""

    def __init__(
        self,
        full_path: str,
        site_meta: dict,
        expected_date: str | None = None,
    ):
        super().__init__(self.noise_param)
        self.full_path = full_path
        self.site_meta = site_meta
        self.expected_date = expected_date
        self._backscatter_scale_factor = 1.0
        self._hex_conversion_params: tuple[int, int, int] = (1, 1, 1)
        self._message_number: int

    def _is_ct25k(self) -> bool:
        if self.instrument is not None and self.instrument.model is not None:
            return "CT25k" in self.instrument.model
        return False

    def _fetch_data_lines(self) -> list:
        """Finds data lines (header + backscatter) from ceilometer file."""
        with open(self.full_path, "rb") as file:
            all_lines = file.readlines()
        return self._screen_invalid_lines(all_lines)

    def _calc_range(self) -> np.ndarray:
        """Calculates range vector from the resolution and number of gates."""
        if self._is_ct25k():
            range_resolution = 30
            n_gates = 256
        else:
            n_gates = int(self.metadata["number_of_gates"])
            range_resolution = int(self.metadata["range_resolution"])
        return np.arange(n_gates) * range_resolution + range_resolution / 2

    def _read_backscatter(self, lines: list) -> np.ndarray:
        """Converts backscatter profile from 2-complement hex to floats."""
        n_chars = self._hex_conversion_params[0]
        n_gates = int(len(lines[0]) / n_chars)
        profiles = np.zeros((len(lines), n_gates), dtype=int)
        ran = range(0, n_gates * n_chars, n_chars)
        for ind, line in enumerate(lines):
            try:
                profiles[ind, :] = [int(line[i : i + n_chars], 16) for i in ran]
            except ValueError:
                logging.warning("Bad value in raw ceilometer data")
        ind = profiles & self._hex_conversion_params[1] != 0
        profiles[ind] -= self._hex_conversion_params[2]
        return profiles.astype(float) / self._backscatter_scale_factor

    def _screen_invalid_lines(self, data_in: list) -> list:
        """Removes empty (and other weird) lines from the list of data."""

        def _filter_lines(data: list) -> list:
            output = []
            for line in data:
                try:
                    output.append(line.decode("utf8"))
                except UnicodeDecodeError:
                    continue
            return output

        def _find_timestamp_line_numbers(data: list) -> list:
            return [n for n, value in enumerate(data) if utils.is_timestamp(value)]

        def _find_correct_dates(data: list, line_numbers: list) -> list:
            return [
                n for n in line_numbers if data[n].strip("-")[:10] == self.expected_date
            ]

        def _find_number_of_data_lines(data: list, timestamp_line_number: int) -> int:
            for i, line in enumerate(data[timestamp_line_number:]):
                if utils.is_empty_line(line):
                    return i
            msg = "Can not parse number of data lines"
            raise RuntimeError(msg)

        def _parse_data_lines(data: list, starting_indices: list) -> list:
            return [
                [
                    data[n + line_number]
                    for n in starting_indices
                    if (n + line_number) < len(data)
                ]
                for line_number in range(number_of_data_lines)
            ]

        valid_lines = _filter_lines(data_in)
        timestamp_line_numbers = _find_timestamp_line_numbers(valid_lines)
        if self.expected_date is not None:
            timestamp_line_numbers = _find_correct_dates(
                valid_lines,
                timestamp_line_numbers,
            )
            if not timestamp_line_numbers:
                raise ValidTimeStampError
        number_of_data_lines = _find_number_of_data_lines(
            valid_lines,
            timestamp_line_numbers[0],
        )
        return _parse_data_lines(valid_lines, timestamp_line_numbers)

    @staticmethod
    def _get_message_number(header_line_1: dict) -> int:
        msg_no = header_line_1["message_number"]
        if len(np.unique(msg_no)) != 1:
            msg = "Error: inconsistent message numbers."
            raise RuntimeError(msg)
        return int(msg_no[0])

    @staticmethod
    def _calc_time(time_lines: list) -> np.ndarray:
        """Returns the time vector as fraction hour."""
        time = [time_to_fraction_hour(line.split()[1]) for line in time_lines]
        return np.array(time)

    @staticmethod
    def _calc_date(time_lines) -> list:
        """Returns the date [yyyy, mm, dd]"""
        return time_lines[0].split()[0].strip("-").split("-")

    @classmethod
    def _handle_metadata(cls, header: list) -> dict:
        meta = cls._concatenate_meta(header)
        meta = cls._remove_meta_duplicates(meta)
        return cls._convert_meta_strings(meta)

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
        strings = (
            "cloud_base_data",
            "measurement_parameters",
            "cloud_amount_data",
        )
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

    def _read_common_header_part(self) -> tuple[list, list]:
        header = []
        data_lines = self._fetch_data_lines()
        self.data["time"] = self._calc_time(data_lines[0])
        self.date = self._calc_date(data_lines[0])
        header.append(self._read_header_line_1(data_lines[1]))
        self._message_number = self._get_message_number(header[0])
        header.append(self._read_header_line_2(data_lines[2]))
        return header, data_lines

    def _read_header_line_1(self, lines: list) -> dict:
        """Reads all first header lines from CT25k and CL ceilometers."""
        fields = (
            "model_id",
            "unit_id",
            "software_level",
            "message_number",
            "message_subclass",
        )
        indices = [1, 3, 4, 6, 7, 8] if self._is_ct25k() else [1, 3, 4, 7, 8, 9]
        values = [split_string(line, indices) for line in lines]
        return values_to_dict(fields, values)

    @staticmethod
    def _read_header_line_2(lines: list) -> dict:
        """Reads the second header line."""
        fields = (
            "detection_status",
            "warning",
            "cloud_base_data",
            "warning_flags",
        )
        values = [[line[0], line[1], line[3:20], line[21:].strip()] for line in lines]
        return values_to_dict(fields, values)


class ClCeilo(VaisalaCeilo):
    """Base class for Vaisala CL31/CL51 ceilometers."""

    noise_param = NoiseParam(noise_min=3.1e-8, noise_smooth_min=1.1e-8)

    def __init__(
        self,
        full_path: str,
        site_meta: dict,
        expected_date: str | None = None,
    ):
        super().__init__(full_path, site_meta, expected_date)
        self._hex_conversion_params = (5, 524288, 1048576)
        self._backscatter_scale_factor = 1e8

    def read_ceilometer_file(self, calibration_factor: float | None = None) -> None:
        """Read all lines of data from the file."""
        header, data_lines = self._read_common_header_part()
        header.append(self._read_header_line_4(data_lines[-3]))
        self.metadata = self._handle_metadata(header)
        self.data["range"] = self._calc_range()
        self.data["beta_raw"] = self._read_backscatter(data_lines[-2])
        self.data["calibration_factor"] = calibration_factor or 1.0
        self.data["beta_raw"] *= self.data["calibration_factor"]
        self.data["zenith_angle"] = np.median(self.metadata["zenith_angle"])
        self._store_ceilometer_info()
        self._sort_time()

    def _sort_time(self) -> None:
        """Sorts timestamps and removes duplicates."""
        time = np.copy(self.data["time"][:])
        ind_sorted = np.argsort(time)
        ind_valid: list[int] = []
        for ind in ind_sorted:
            if time[ind] not in time[ind_valid]:
                ind_valid.append(ind)
        n_time = len(time)
        for key, array in self.data.items():
            if not hasattr(array, "shape"):
                continue
            if array.ndim == 1 and array.shape[0] == n_time:
                self.data[key] = self.data[key][ind_valid]
            if array.ndim == 2 and array.shape[0] == n_time:
                self.data[key] = self.data[key][ind_valid, :]

    def _store_ceilometer_info(self) -> None:
        n_gates = self.data["beta_raw"].shape[1]
        if n_gates < 1540:
            self.instrument = instruments.CL31
        else:
            self.instrument = instruments.CL51

    def _read_header_line_3(self, lines: list) -> dict:
        if self._message_number != 2:
            msg = f"Unsupported message number: {self._message_number}"
            raise RuntimeError(msg)
        keys = ("cloud_detection_status", "cloud_amount_data")
        values = [[line[0:3], line[3:].strip()] for line in lines]
        return values_to_dict(keys, values)

    @staticmethod
    def _read_header_line_4(lines: list) -> dict:
        keys = (
            "scale",
            "range_resolution",
            "number_of_gates",
            "laser_energy",
            "laser_temperature",
            "window_transmission",
            "zenith_angle",
            "background_light",
            "measurement_parameters",
            "backscatter_sum",
        )
        values = [line.split() for line in lines]
        return values_to_dict(keys, values)


class Ct25k(VaisalaCeilo):
    """Class for Vaisala CT25k ceilometer.

    References
    ----------
        https://www.manualslib.com/manual/1414094/Vaisala-Ct25k.html

    """

    noise_param = NoiseParam(noise_min=0.7e-7, noise_smooth_min=1.2e-8)

    def __init__(
        self,
        input_file: str,
        site_meta: dict,
        expected_date: str | None = None,
    ):
        super().__init__(input_file, site_meta, expected_date)
        self._hex_conversion_params = (4, 32768, 65536)
        self._backscatter_scale_factor = 1e7
        self.instrument = instruments.CT25K

    def read_ceilometer_file(self, calibration_factor: float | None = None) -> None:
        """Read all lines of data from the file."""
        header, data_lines = self._read_common_header_part()
        header.append(self._read_header_line_3(data_lines[3]))
        self.metadata = self._handle_metadata(header)
        self.data["range"] = self._calc_range()
        hex_profiles = self._parse_hex_profiles(data_lines[4:20])
        self.data["beta_raw"] = self._read_backscatter(hex_profiles)
        self.data["calibration_factor"] = calibration_factor or 1.0
        self.data["beta_raw"] *= self.data["calibration_factor"]
        self.data["zenith_angle"] = np.median(self.metadata["zenith_angle"])

    @staticmethod
    def _parse_hex_profiles(lines: list) -> list:
        """Collects ct25k profiles into list (one profile / element)."""
        n_profiles = len(lines[0])
        return [
            "".join([lines[m][n][3:].strip() for m in range(16)])
            for n in range(n_profiles)
        ]

    def _read_header_line_3(self, lines: list) -> dict:
        if self._message_number in (1, 3, 6):
            msg = f"Unsupported message number: {self._message_number}"
            raise RuntimeError(msg)
        keys = (
            "measurement_mode",
            "laser_energy",
            "laser_temperature",
            "receiver_sensitivity",
            "window_contamination",
            "zenith_angle",
            "background_light",
            "measurement_parameters",
            "backscatter_sum",
        )
        values = [line.split() for line in lines]
        keys_out = ("scale", *keys) if len(values[0]) == 10 else keys
        return values_to_dict(keys_out, values)


def split_string(string: str, indices: list) -> list:
    """Splits string between indices.

    Notes
    -----
        It is possible to skip characters from the beginning and end of the
        string but not from the middle.

    Examples
    --------
        >>> s = 'abcde'
        >>> indices = [1, 2, 4]
        >>> split_string(s, indices)
        ['b', 'cd']

    """
    return [string[n:m] for n, m in itertools.pairwise(indices)]


def values_to_dict(keys: tuple, values: list) -> dict:
    """Converts list elements to dictionary.

    Examples
    --------
        >>> keys = ('a', 'b')
        >>> values = [[1, 2], [1, 2], [1, 2], [1, 2]]
        >>> values_to_dict(keys, values)
        {'a': array([1, 1, 1, 1]), 'b': array([2, 2, 2, 2])}

    """
    out = {}
    for i, key in enumerate(keys):
        out[key] = np.array([x[i] for x in values if len(x) == len(keys)])
    return out


def time_to_fraction_hour(time: str) -> float:
    """Returns time (hh:mm:ss) as fraction hour"""
    hour, minute, sec = time.split(":")
    return int(hour) + (int(minute) * SEC_IN_MINUTE + int(sec)) / SEC_IN_HOUR
