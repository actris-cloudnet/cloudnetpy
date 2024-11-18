"""Module for reading Radiometrics MP3014 microwave radiometer data."""

import csv
import datetime
import logging
import os
import re
from operator import attrgetter
from typing import Any, NamedTuple

import numpy as np

from cloudnetpy import output, utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.exceptions import InconsistentDataError, ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.metadata import MetaData


def radiometrics2nc(
    full_path: str,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | datetime.date | None = None,
) -> str:
    """Converts Radiometrics .csv file into Cloudnet Level 1b netCDF file.

    Args:
        full_path: Input file name or folder containing multiple input files.
        output_file: Output file name, e.g. 'radiometrics.nc'.
        site_meta: Dictionary containing information about the site and instrument.
            Required key value pairs are `name` and `altitude` (metres above mean
            sea level).
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.instruments import radiometrics2nc
        >>> site_meta = {'name': 'Soverato', 'altitude': 21}
        >>> radiometrics2nc('radiometrics.csv', 'radiometrics.nc', site_meta)

    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)

    if os.path.isdir(full_path):
        valid_filenames = utils.get_sorted_filenames(full_path, ".csv")
    else:
        valid_filenames = [full_path]

    objs = []
    for filename in valid_filenames:
        obj = Radiometrics(filename)
        obj.read_raw_data()
        obj.read_data()
        objs.append(obj)

    radiometrics = RadiometricsCombined(objs, site_meta)
    radiometrics.screen_time(date)
    radiometrics.time_to_fractional_hours()
    radiometrics.data_to_cloudnet_arrays()
    radiometrics.add_meta()
    if radiometrics.date is None:
        msg = "Failed to find valid timestamps from Radiometrics file(s)."
        raise ValidTimeStampError(msg)
    attributes = output.add_time_attribute(ATTRIBUTES, radiometrics.date)
    output.update_attributes(radiometrics.data, attributes)
    return output.save_level1b(radiometrics, output_file, uuid)


class Record(NamedTuple):
    row_number: int
    block_type: int
    block_index: int
    timestamp: datetime.datetime
    values: dict[str, Any]


class Radiometrics:
    """Reader for level 2 files of Radiometrics microwave radiometers.

    References:
        Radiometrics (2008). Profiler Operator's Manual: MP-3000A, MP-2500A,
        MP-1500A, MP-183A.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.raw_data: list[Record] = []
        self.data: dict = {}
        self.instrument = instruments.RADIOMETRICS
        self.ranges: list[str] = []

    def read_raw_data(self) -> None:
        """Reads Radiometrics raw data."""
        record_columns = {}
        unknown_record_types = set()
        rows = []
        with open(self.filename, encoding="utf8") as infile:
            reader = csv.reader(infile, skipinitialspace=True)
            for row in reader:
                if row[0] == "Record":
                    if row[1] != "Date/Time":
                        msg = "Unexpected header in Radiometrics file"
                        raise RuntimeError(msg)
                    record_type = int(row[2])
                    columns = row[3:]
                    record_columns[record_type] = columns
                    if record_type in (10, 400):
                        self.ranges = [
                            column
                            for column in columns
                            if re.fullmatch(r"\d+\.\d+", column)
                        ]
                else:
                    record_type = int(row[2])
                    block_type = record_type // 10 * 10
                    block_index = record_type - block_type - 1
                    column_names = record_columns.get(block_type)
                    if column_names is None:
                        if record_type not in unknown_record_types:
                            logging.info("Skipping unknown record type %d", record_type)
                            unknown_record_types.add(record_type)
                        continue
                    record = Record(
                        row_number=int(row[0]),
                        timestamp=_parse_datetime(row[1]),
                        block_type=block_type,
                        block_index=block_index,
                        values=dict(zip(column_names, row[3:], strict=True)),
                    )
                    rows.append(record)

        self.raw_data = sorted(rows, key=attrgetter("row_number"))

    def read_data(self) -> None:
        """Reads values."""
        times = []
        lwps = []
        iwvs = []
        irts = []
        temps = []
        rhs = []
        ahs = []
        block_titles = {}
        for record in self.raw_data:
            if record.block_type == 100:
                block_type = int(record.values["Record Type"]) - 1
                title = record.values["Title"]
                block_titles[block_type] = title
            if title := block_titles.get(record.block_type + record.block_index):
                if title == "Temperature (K)":
                    temps.append(
                        [float(record.values[column]) for column in self.ranges]
                    )
                elif title == "Relative Humidity (%)":
                    rhs.append([float(record.values[column]) for column in self.ranges])
                elif title == "Vapor Density (g/m^3)":
                    ahs.append([float(record.values[column]) for column in self.ranges])
            elif record.block_type == 10:
                if record.block_index == 0:
                    lwp = record.values["Lqint(mm)"]
                    iwv = record.values["Vint(cm)"]
                    irt = record.values["Tir(K)"]
                    times.append(record.timestamp)
                    lwps.append(float(lwp))
                    iwvs.append(float(iwv))
                    irts.append([float(irt)])
                    temps.append(
                        [float(record.values[column]) for column in self.ranges]
                    )
                elif record.block_index == 1:
                    ahs.append([float(record.values[column]) for column in self.ranges])
                elif record.block_index == 2:
                    rhs.append([float(record.values[column]) for column in self.ranges])
            elif record.block_type == 200:
                irt = record.values["Tir(K)"]
                irts.append([float(irt)])
            elif record.block_type == 300:
                lwp = record.values["Int. Liquid(mm)"]
                iwv = record.values["Int. Vapor(cm)"]
                times.append(record.timestamp)
                lwps.append(float(lwp))
                iwvs.append(float(iwv))
        n_time = len(times)
        self.data["time"] = np.array(times, dtype="datetime64[s]")
        self.data["lwp"] = np.array(lwps)  # mm => kg m-2
        self.data["iwv"] = np.array(iwvs) * 10  # cm => kg m-2
        self.data["irt"] = np.array(irts[:n_time])
        self.data["temperature"] = np.array(temps[:n_time])
        self.data["relative_humidity"] = np.array(rhs[:n_time]) / 100  # % => 1
        self.data["absolute_humidity"] = (
            np.array(ahs[:n_time]) / 1000
        )  # g m-3 => kg m-3


class RadiometricsCombined:
    site_meta: dict
    data: dict
    date: datetime.date | None
    instrument: instruments.Instrument

    def __init__(self, objs: list[Radiometrics], site_meta: dict):
        self.site_meta = site_meta
        self.data = {}
        self.date = None
        for obj in objs:
            if obj.ranges != objs[0].ranges:
                msg = "Inconsistent range between files"
                raise InconsistentDataError(msg)
            for key in obj.data:
                self.data = utils.append_data(self.data, key, obj.data[key])
        ranges = [float(x) for x in objs[0].ranges]
        self.data["range"] = np.array(ranges) * 1000  # m => km
        self.data["height"] = self.data["range"] + self.site_meta["altitude"]
        self.instrument = instruments.RADIOMETRICS

    def screen_time(self, expected_date: datetime.date | None) -> None:
        """Screens timestamps."""
        if expected_date is None:
            self.date = self.data["time"][0].astype(object).date()
            return
        self.date = expected_date
        valid_mask = self.data["time"].astype("datetime64[D]") == self.date
        if np.count_nonzero(valid_mask) == 0:
            raise ValidTimeStampError
        for key in self.data:
            if key in ("range", "height"):
                continue
            self.data[key] = self.data[key][valid_mask]

    def time_to_fractional_hours(self) -> None:
        base = self.data["time"][0].astype("datetime64[D]")
        self.data["time"] = (self.data["time"] - base) / np.timedelta64(1, "h")

    def data_to_cloudnet_arrays(self) -> None:
        """Converts arrays to CloudnetArrays."""
        for key, array in self.data.items():
            dimensions = (
                ("time", "range")
                if key in ("temperature", "relative_humidity", "absolute_humidity")
                else None
            )
            self.data[key] = CloudnetArray(array, key, dimensions=dimensions)

    def add_meta(self) -> None:
        """Adds some metadata."""
        valid_keys = ("latitude", "longitude", "altitude")
        for key, value in self.site_meta.items():
            name = key.lower()
            if name in valid_keys:
                self.data[name] = CloudnetArray(float(value), key)


def _parse_datetime(text: str) -> datetime.datetime:
    date, time = text.split()
    month, day, year = map(int, date.split("/"))
    hour, minute, second = map(int, time.split(":"))
    if year < 100:
        year += 2000
    return datetime.datetime(
        year,
        month,
        day,
        hour,
        minute,
        second,
    )


ATTRIBUTES = {
    "irt": MetaData(
        long_name="Infrared brightness temperatures",
        units="K",
    ),
}
