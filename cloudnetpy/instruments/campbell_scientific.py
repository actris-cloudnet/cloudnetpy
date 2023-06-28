import logging
import re
from datetime import datetime

import numpy as np

from cloudnetpy import utils
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.ceilometer import Ceilometer


class Cs135(Ceilometer):
    def __init__(
        self, full_path: str, site_meta: dict, expected_date: str | None = None
    ):
        super().__init__()
        self.full_path = full_path
        self.site_meta = site_meta
        self.expected_date = expected_date
        self.data = {}
        self.metadata = {}
        self.instrument = instruments.CS135

    def read_ceilometer_file(self, calibration_factor: float | None = None) -> None:
        with open(self.full_path, mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        timestamps, profiles, scales, tilt_angles = [], [], [], []
        range_resolution, n_gates = 0, 0
        for i, line in enumerate(lines):
            line_splat = line.strip().split(",")
            if is_timestamp(line_splat[0]):
                timestamp = datetime.strptime(line_splat[0], "%Y-%m-%dT%H:%M:%S.%f")
                try:
                    self._check_timestamp(timestamp)
                except ValidTimeStampError:
                    continue
                timestamps.append(timestamp)
                _line1 = line_splat[1]
                if len(_line1) != 11:
                    raise NotImplementedError("Unknown message number")
                if (msg_no := _line1[-4:-1]) != "002":
                    raise NotImplementedError(
                        f"Message number {msg_no} not implemented"
                    )
                _line3 = lines[i + 2].strip().split(" ")
                scale, range_resolution, n_gates, tilt_angle = [
                    int(_line3[ind]) for ind in [0, 1, 2, 5]
                ]
                scales.append(scale)
                tilt_angles.append(tilt_angle)
                _line4 = lines[i + 3].strip()
                profiles.append(_hex2backscatter(_line4, n_gates))

        if len(timestamps) == 0:
            raise ValidTimeStampError("No valid timestamps found in the file")
        array = self._handle_large_values(np.array(profiles))
        self.data["beta_raw"] = _scale_backscatter(array, scales)
        if calibration_factor is None:
            calibration_factor = 1.0
        self.data["beta_raw"] *= calibration_factor
        self.data["calibration_factor"] = calibration_factor
        self.data["range"] = (
            np.arange(n_gates) * range_resolution + range_resolution / 2
        )
        self.data["time"] = utils.datetime2decimal_hours(timestamps)
        self.data["zenith_angle"] = np.median(tilt_angles)

    def _check_timestamp(self, timestamp: datetime):
        timestamp_components = str(timestamp.date()).split("-")
        if self.expected_date is not None:
            if timestamp_components != self.expected_date.split("-"):
                raise ValidTimeStampError
        if not self.date:
            self.date = timestamp_components
        assert timestamp_components == self.date

    @staticmethod
    def _handle_large_values(array: np.ndarray) -> np.ndarray:
        ind = np.where(array > 524287)
        if ind[0].size > 0:
            array[ind] -= 1048576
        return array


def is_timestamp(timestamp: str) -> bool:
    """Tests if the input string is formatted as -yyyy-mm-dd hh:mm:ss"""
    reg_exp = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{6}")
    if reg_exp.match(timestamp) is not None:
        return True
    return False


def _hex2backscatter(data: str, n_gates: int):
    """Converts hex string to backscatter values."""
    n_chars = 5
    return [
        int(data[i : i + n_chars], 16) for i in range(0, n_gates * n_chars, n_chars)
    ]


def _scale_backscatter(data: np.ndarray, scales: list) -> np.ndarray:
    """Scales backscatter values."""
    unit_conversion_factor = 1e-8
    scales_array = np.array(scales)
    ind = np.where(scales_array != 100)
    if ind[0].size > 0:
        logging.info(f"{ind[0].size} profiles have not 100% scaling")
        data[ind, :] *= scales_array[ind] / 100
    return data * unit_conversion_factor
