import binascii
import datetime
import re
from typing import NamedTuple

import numpy as np

from cloudnetpy import utils
from cloudnetpy.exceptions import InconsistentDataError, ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.ceilometer import Ceilometer


def _date_format_to_regex(fmt: bytes) -> bytes:
    """Converts a date format string to a regex pattern."""
    mapping = {
        b"%Y": rb"\d{4}",
        b"%m": rb"0[1-9]|1[0-2]",
        b"%d": rb"0[1-9]|[12]\d|3[01]",
        b"%H": rb"[01]\d|2[0-3]",
        b"%M": rb"[0-5]\d",
        b"%S": rb"[0-5]\d",
        b"%f": rb"\d{6}",
    }
    pattern = re.escape(fmt)
    for key, value in mapping.items():
        pattern = pattern.replace(
            re.escape(key), b"(?P<" + key[1:] + b">" + value + b")"
        )
    return pattern


FORMATS = [
    re.compile(_date_format_to_regex(fmt))
    for fmt in [
        b"%Y-%m-%dT%H:%M:%S.%f,",
        b"%%% %Y/%m/%d %H:%M:%S %%%\n",
    ]
]


class Cs135(Ceilometer):
    def __init__(
        self,
        full_path: str,
        site_meta: dict,
        expected_date: str | None = None,
    ):
        super().__init__()
        self.full_path = full_path
        self.site_meta = site_meta
        self.expected_date = expected_date
        self.data = {}
        self.metadata = {}
        self.instrument = instruments.CS135

    def read_ceilometer_file(self, calibration_factor: float | None = None) -> None:
        with open(self.full_path, mode="rb") as f:
            content = f.read()
        timestamps = []
        profiles = []
        tilt_angles = []
        range_resolutions = []

        for fmt in FORMATS:
            parts = re.split(fmt, content)
            for i in range(1, len(parts), fmt.groups + 1):
                timestamp = datetime.datetime(
                    int(parts[i + fmt.groupindex["Y"] - 1]),
                    int(parts[i + fmt.groupindex["m"] - 1]),
                    int(parts[i + fmt.groupindex["d"] - 1]),
                    int(parts[i + fmt.groupindex["H"] - 1]),
                    int(parts[i + fmt.groupindex["M"] - 1]),
                    int(parts[i + fmt.groupindex["S"] - 1]),
                    int(parts[i + fmt.groupindex["f"] - 1])
                    if "f" in fmt.groupindex
                    else 0,
                    tzinfo=datetime.timezone.utc,
                )
                try:
                    self._check_timestamp(timestamp)
                except ValidTimeStampError:
                    continue
                try:
                    message = _read_message(parts[i + fmt.groups])
                except InvalidMessageError:
                    continue
                profile = (message.data[:-2] * 1e-8) * (message.scale / 100)
                timestamps.append(timestamp)
                profiles.append(profile)
                tilt_angles.append(message.tilt_angle)
                range_resolutions.append(message.range_resolution)

        if len(timestamps) == 0:
            msg = "No valid timestamps found in the file"
            raise ValidTimeStampError(msg)
        range_resolution = range_resolutions[0]
        n_gates = len(profiles[0])
        if any(res != range_resolution for res in range_resolutions):
            msg = "Inconsistent range resolution"
            raise InconsistentDataError(msg)
        if any(len(profile) != n_gates for profile in profiles):
            msg = "Inconsistent number of gates"
            raise InconsistentDataError(msg)

        self.data["beta_raw"] = np.array(profiles)
        if calibration_factor is None:
            calibration_factor = 1.0
        self.data["beta_raw"] *= calibration_factor
        self.data["calibration_factor"] = calibration_factor
        self.data["range"] = (
            np.arange(n_gates) * range_resolution + range_resolution / 2
        )
        self.data["time"] = utils.datetime2decimal_hours(timestamps)
        self.data["zenith_angle"] = np.median(tilt_angles)

    def _check_timestamp(self, timestamp: datetime.datetime) -> None:
        timestamp_components = str(timestamp.date()).split("-")
        if (
            self.expected_date is not None
            and timestamp_components != self.expected_date.split("-")
        ):
            raise ValidTimeStampError
        if not self.date:
            self.date = timestamp_components
        if timestamp_components != self.date:
            msg = "Inconsistent dates in the file"
            raise RuntimeError(msg)


class Message(NamedTuple):
    scale: int
    range_resolution: int
    laser_pulse_energy: int
    laser_temperature: int
    tilt_angle: int
    background_light: int
    pulse_quantity: int
    sample_rate: int
    data: np.ndarray


class InvalidMessageError(Exception):
    pass


def _read_message(message: bytes) -> Message:
    end_idx = message.index(3)
    content = message[1 : end_idx + 1]
    expected_checksum = int(message[end_idx + 1 : end_idx + 5], 16)
    actual_checksum = _crc16(content)
    if expected_checksum != actual_checksum:
        msg = (
            "Invalid checksum: "
            f"expected {expected_checksum:04x}, "
            f"got {actual_checksum:04x}"
        )
        raise InvalidMessageError(msg)
    lines = message[1 : end_idx - 1].splitlines()
    n_lines = len(lines) + 1
    n_first = len(lines[0]) + 1
    if n_first != 11:
        msg = f"Expected 11 characters in first line, got {n_first}"
        raise NotImplementedError(msg)
    msg_no = lines[0][-4:-1]
    if msg_no == b"002":
        if n_lines != 5:
            msg = f"Expected 5 lines, got {len(lines)}"
            raise InvalidMessageError(msg)
        scale, res, n, energy, lt, ti, bl, pulse, rate, _sum = map(
            int, lines[2].split()
        )
        data = _read_backscatter(lines[3].strip(), n)
        return Message(scale, res, energy, lt, ti, bl, pulse, rate, data)
    if msg_no == b"004":
        if n_lines != 6:
            msg = f"Expected 6 lines, got {len(lines)}"
            raise InvalidMessageError(msg)
        scale, res, n, energy, lt, ti, bl, pulse, rate, _sum = map(
            int, lines[3].split()
        )
        data = _read_backscatter(lines[4].strip(), n)
        return Message(scale, res, energy, lt, ti, bl, pulse, rate, data)
    msg = f"Message number {msg_no.decode()} not implemented"
    raise NotImplementedError(msg)


def _read_backscatter(data: bytes, n_gates: int) -> np.ndarray:
    """Read backscatter values from hex-encoded two's complement values."""
    n_chars = 5
    n_bits = n_chars * 4
    limit = (1 << (n_bits - 1)) - 1
    offset = 1 << n_bits
    out = np.array(
        [int(data[i : i + n_chars], 16) for i in range(0, n_gates * n_chars, n_chars)],
    )
    out[out > limit] -= offset
    return out


def _crc16(data: bytes) -> int:
    """Compute checksum similar to CRC-16-CCITT."""
    return binascii.crc_hqx(data, 0xFFFF) ^ 0xFFFF
