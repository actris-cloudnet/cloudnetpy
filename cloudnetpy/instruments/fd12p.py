import datetime
import logging
import math
import re
from os import PathLike
from uuid import UUID

import numpy as np
from numpy import ma

from cloudnetpy import output
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CSVFile
from cloudnetpy.metadata import MetaData


def fd12p2nc(
    input_file: str | PathLike,
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
):
    """Converts Vaisala FD12P into Cloudnet Level 1b netCDF file.

    Args:
        input_file: Filename of input file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD or datetime.date object.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.
    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    if isinstance(uuid, str):
        uuid = UUID(uuid)
    fd12p = FD12P(site_meta)
    fd12p.parse_input_file(input_file, date)
    fd12p.add_data()
    fd12p.add_date()
    fd12p.screen_all_masked()
    fd12p.sort_timestamps()
    fd12p.remove_duplicate_timestamps()
    fd12p.convert_units()
    fd12p.normalize_cumulative_amount("precipitation_amount")
    fd12p.normalize_cumulative_amount("snowfall_amount")
    fd12p.add_site_geolocation()
    attributes = output.add_time_attribute(ATTRIBUTES, fd12p.date)
    output.update_attributes(fd12p.data, attributes)
    return output.save_level1b(fd12p, output_file, uuid)


class FD12P(CSVFile):
    def __init__(self, site_meta: dict):
        super().__init__(site_meta)
        self.instrument = instruments.FD12P
        self._data = {
            key: []
            for key in (
                "time",
                "visibility",
                "synop_WaWa",
                "precipitation_rate",
                "precipitation_amount",
                "snowfall_amount",
            )
        }

    def parse_input_file(
        self, filename: str | PathLike, expected_date: datetime.date | None = None
    ):
        # In Lindenberg, format is date and time followed by Message 2 without
        # non-printable characters.
        with open(filename) as file:
            invalid_lines = 0
            for line in file:
                try:
                    columns = line.split()
                    if len(columns) != 13:
                        msg = "Invalid column count"
                        raise ValueError(msg)  # noqa: TRY301
                    date = _parse_date(columns[0])
                    time = _parse_time(columns[1])
                    visibility = _parse_int(columns[4])
                    synop = _parse_int(columns[7])
                    p_rate = _parse_float(columns[10])  # mm/h
                    p_amount = _parse_float(columns[11])  # mm
                    s_amount = _parse_int(columns[12])  # mm
                    self._data["time"].append(datetime.datetime.combine(date, time))
                    self._data["visibility"].append(visibility)
                    self._data["synop_WaWa"].append(synop)
                    self._data["precipitation_rate"].append(p_rate)
                    self._data["precipitation_amount"].append(p_amount)
                    self._data["snowfall_amount"].append(s_amount)
                except ValueError:
                    invalid_lines += 1
                    continue
        if invalid_lines:
            logging.info("Skipped %d lines", invalid_lines)
        for key in ("visibility", "synop_WaWa", "snowfall_amount"):
            values = np.array(
                [0 if x is math.nan else x for x in self._data[key]], dtype=np.int32
            )
            mask = np.array([x is math.nan for x in self._data[key]])
            self._data[key] = ma.array(values, mask=mask)
        self._data["snowfall_amount"] = self._data["snowfall_amount"].astype(np.float32)
        if expected_date:
            self._data["time"] = [
                d for d in self._data["time"] if d.date() == expected_date
            ]
        if not self._data["time"]:
            raise ValidTimeStampError

    def convert_units(self) -> None:
        precipitation_rate = self.data["precipitation_rate"][:]
        self.data["precipitation_rate"].data = (
            precipitation_rate / 3600 / 1000
        )  # mm/h -> m/s
        for key in ("precipitation_amount", "snowfall_amount"):
            self.data[key].data = self.data[key][:] / 1000  # mm -> m

    def screen_all_masked(self) -> None:
        is_valid = np.ones_like(self.data["time"][:], dtype=np.bool)
        for key in self.data:
            if key == "time":
                continue
            is_valid &= ma.getmaskarray(self.data[key][:])
        self.screen_time_indices(~is_valid)


def _parse_date(date: str) -> datetime.date:
    match = re.fullmatch(r"(?P<day>\d{2})\.(?P<month>\d{2})\.(?P<year>\d{4})", date)
    if match is None:
        msg = f"Invalid date: {date}"
        raise ValueError(msg)
    return datetime.date(int(match["year"]), int(match["month"]), int(match["day"]))


def _parse_time(time: str) -> datetime.time:
    match = re.fullmatch(
        r"(?P<hour>\d{2}):(?P<minute>\d{2})(:(?P<second>\d{2}))?", time
    )
    if match is None:
        msg = f"Invalid time: {time}"
        raise ValueError(msg)
    return datetime.time(
        int(match["hour"]),
        int(match["minute"]),
        int(match["second"]) if match["second"] is not None else 0,
    )


def _parse_int(value: str) -> float:
    if "/" in value:
        return math.nan
    return int(value)


def _parse_float(value: str) -> float:
    if "/" in value:
        return math.nan
    return float(value)


ATTRIBUTES = {
    "visibility": MetaData(
        long_name="Meteorological optical range (MOR) visibility",
        units="m",
        standard_name="visibility_in_air",
    ),
    "precipitation_rate": MetaData(
        long_name="Precipitation rate",
        standard_name="lwe_precipitation_rate",
        units="m s-1",
    ),
    "precipitation_amount": MetaData(
        long_name="Precipitation amount",
        standard_name="lwe_thickness_of_precipitation_amount",
        units="m",
        comment="Cumulated precipitation since 00:00 UTC",
    ),
    "snowfall_amount": MetaData(
        long_name="Snowfall amount",
        units="m",
        standard_name="thickness_of_snowfall_amount",
        comment="Cumulated snow since 00:00 UTC",
    ),
    "synop_WaWa": MetaData(long_name="Synop code WaWa", units="1"),
}
