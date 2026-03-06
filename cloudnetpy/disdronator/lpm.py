import datetime
from os import PathLike
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from cloudnetpy.disdronator.utils import convert_to_numpy

LpmOutput: TypeAlias = tuple[list, dict[int, list]]

# fmt: off
INT_KEYS = {
    2, 3, 7, 8, 11, 12, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 524,
}
FLOAT_KEYS = {10, 14, 15, 16, 17, 19, 21, 46, 521, 522, 523}
FILL_VALUES = {
    18: 99999,
    19: -9.9,
    46: 99999,
    47: 999,
    48: 9999,
    49: 9999,
    50: 9999,
    521: 99999,
    522: 99999,
    523: 9999,
    524: 999,
}
Dlow = np.array([
    0.125, 0.250, 0.375, 0.500, 0.750, 1.000, 1.250, 1.500, 1.750, 2.000, 2.500,
    3.000, 3.500, 4.000, 4.500, 5.000, 5.500, 6.000, 6.500, 7.000, 7.500,
    8.000,
])
Dspr = np.array([
    0.125, 0.125, 0.125, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.500, 0.500,
    0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500,
    0.500,
])
Vlow = np.array([
    0.000, 0.200, 0.400, 0.600, 0.800, 1.000, 1.400, 1.800, 2.200, 2.600, 3.000,
    3.400, 4.200, 5.000, 5.800, 6.600, 7.400, 8.200, 9.000, 10.000,
])
Vspr = np.array([
    0.200, 0.200, 0.200, 0.200, 0.200, 0.400, 0.400, 0.400, 0.400, 0.400, 0.400,
    0.800, 0.800, 0.800, 0.800, 0.800, 0.800, 0.800, 1.000, 10.000,
])
# fmt: on

Dmid = Dlow + Dspr / 2
Vmid = Vlow + Vspr / 2
A = (20 / 1000) * (228 / 1000)  # TODO: AU


def _read_telegram(telegram: str, data: dict[int, list]) -> None:
    telegram = telegram.strip().rstrip(";")
    values = telegram.split(";")
    if len(values) not in (520, 524):
        msg = "Invalid telegram length"
        raise ValueError(msg)
    for i, value in enumerate(values[:-1]):
        no = i + 2
        parsed: datetime.date | datetime.time | int | float | str
        if no == 5:
            parsed = datetime.datetime.strptime(value, "%d.%m.%y").date()
        elif no == 6:
            parsed = datetime.datetime.strptime(value, "%H:%M:%S").time()
        elif no in INT_KEYS or 81 <= no <= 520:
            parsed = int(value)
        elif no in FLOAT_KEYS:
            parsed = float(value)
        else:
            parsed = value
        if no not in data:
            data[no] = []
        data[no].append(parsed)


def _read_pyatmoslogger(filename: str | PathLike) -> LpmOutput:
    time = []
    data: dict = {}
    with open(filename, errors="ignore") as f:
        f.readline()
        for line in f:
            timestamp, telegram = line.split(";", maxsplit=1)
            try:
                _read_telegram(telegram, data)
                time.append(datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass
    return time, data


def _read_lampedusa(filename: str | PathLike) -> LpmOutput:
    time = []
    data: dict = {}
    with open(filename) as f:
        _, _, _, _ = f.readline(), f.readline(), f.readline(), f.readline()
        for line in f:
            cols = [col.strip('"') for col in line.strip().split(",")]
            try:
                _read_telegram(cols[2], data)
                time.append(datetime.datetime.strptime(cols[0], "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                pass
    return time, data


def _read_raw(filename: str | PathLike) -> LpmOutput:
    time = []
    data: dict = {}
    with open(filename) as f:
        for line in f:
            try:
                _read_telegram(line, data)
                time.append(datetime.datetime.combine(data[5][-1], data[6][-1]))
            except ValueError:
                pass
    return time, data


def read_lpm(filename: str | PathLike) -> tuple[npt.NDArray, dict[int, npt.NDArray]]:
    with open(filename, "rb") as f:
        head = f.read(50)
    if head.lower().startswith(b"datetime [utc]"):
        time, data = _read_pyatmoslogger(filename)
    elif b"TOA5" in head:
        time, data = _read_lampedusa(filename)
    else:
        time, data = _read_raw(filename)
    return np.ndarray(time), convert_to_numpy(data, FILL_VALUES)
