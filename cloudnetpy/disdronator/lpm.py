import datetime
from os import PathLike
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

from cloudnetpy.disdronator.process import DisdroL1
from cloudnetpy.disdronator.utils import convert_to_numpy

LpmOutput: TypeAlias = tuple[list, dict[int, list]]


def _read_telegram(telegram: str, data: dict[int, list]) -> None:
    telegram = telegram.lstrip("\x02").rstrip(";\r\n\x03 ")
    values = telegram.split(";")
    # 520 = no weather data, 524 = weather data
    if len(values) not in (520, 524):
        msg = "Invalid telegram length"
        raise ValueError(msg)
    # There's something wrong with checksum validation when there's weather data
    # (only one in L'Aquila).
    if len(values) == 520:
        data_chars = b"\x02" + telegram[:-2].encode() + b";\r\n\x03"
        checksum = -np.sum(np.frombuffer(data_chars, dtype=np.int8)) & 0xFF
        checksum_hex = f"{checksum:02X}"
        if checksum_hex != telegram[-2:]:
            msg = "Invalid telegram checksum"
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
    return np.array(time), convert_to_numpy(data, FILL_VALUES)


def read_lpm_l1(
    time: npt.NDArray, l0: dict[int, npt.NDArray], au: int | None = None
) -> DisdroL1:
    area_nom = 4600 * 1000 / au if au is not None else 4560
    area_eff = area_nom * (1 - D / (2 * LASER_WIDTH))
    data_raw = np.stack([l0[i] for i in range(81, 521)], axis=1).reshape(
        (len(time), len(D), len(V))
    )
    return DisdroL1(
        diameter=D,
        diameter_spread=D_SPREAD,
        diameter_bins=D_BINS,
        velocity=V,
        velocity_spread=V_SPREAD,
        velocity_bins=V_BINS,
        time=time,
        interval=np.repeat(60, len(time)),
        area_nom=area_nom,
        area_eff=area_eff,
        data_raw=data_raw,
    )


# fmt: off
INT_KEYS = {
    2, 7, 8, 11, 12, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 524,
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
D = np.array([
    0.1875, 0.3125, 0.4375, 0.6250, 0.8750, 1.1250, 1.3750, 1.6250, 1.8750,
    2.2500, 2.7500, 3.2500, 3.7500, 4.2500, 4.7500, 5.2500, 5.7500, 6.2500,
    6.7500, 7.2500, 7.7500, 8.2500
])
D_BINS = np.array([
    0.125, 0.250, 0.375, 0.500, 0.750, 1.000, 1.250, 1.500, 1.750, 2.000, 2.500,
    3.000, 3.500, 4.000, 4.500, 5.000, 5.500, 6.000, 6.500, 7.000, 7.500,
    8.000, 8.500
])
D_SPREAD = np.array([
    0.125, 0.125, 0.125, 0.250, 0.250, 0.250, 0.250, 0.250, 0.250, 0.500, 0.500,
    0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500, 0.500,
    0.500
])
V = np.array([
    0.1, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2.0, 2.4, 2.8, 3.2, 3.8, 4.6, 5.4, 6.2,
    7.0, 7.8, 8.6, 9.5, 15.0
])
V_BINS = np.array([
    0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 4.2, 5.0, 5.8,
    6.6, 7.4, 8.2, 9.0, 10.0, 20.0
])
V_SPREAD = np.array([
    0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8,
    0.8, 0.8, 0.8, 1.0, 10.0,
])
LASER_WIDTH = 20
# fmt: on
