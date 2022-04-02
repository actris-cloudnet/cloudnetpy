"""Module for reading Radiometrics MP3014 microwave radiometer data."""
import csv
from typing import List, Optional

import numpy as np

from cloudnetpy import CloudnetArray, output
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import instruments


def radiometrics2nc(
    full_path: str,
    output_file: str,
    site_meta: dict,
    uuid: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """Converts Radiometrics .csv file into Cloudnet Level 1b netCDF file.

    Args:
        full_path: Input file name.
        output_file: Output file name, e.g. 'radiometrics.nc'.
        site_meta: Dictionary containing information about the site and instrument.
            Required key value pairs are `name` and `altitude` (metres above mean sea level).
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Examples:
        >>> from cloudnetpy.instruments import radiometrics2nc
        >>> site_meta = {'name': 'Soverato', 'altitude': 21}
        >>> radiometrics2nc('radiometrics.csv', 'radiometrics.nc', site_meta)

    """
    radiometrics = Radiometrics(full_path, site_meta)
    radiometrics.read_raw_data()
    radiometrics.read_lwp()
    radiometrics.read_timestamps()
    radiometrics.screen_time(date)
    radiometrics.data_to_cloudnet_arrays()
    radiometrics.add_meta()
    attributes = output.add_time_attribute({}, radiometrics.date)
    output.update_attributes(radiometrics.data, attributes)
    uuid = output.save_level1b(radiometrics, output_file, uuid)
    return uuid


class Radiometrics:
    """Class for Radiometrics MWR"""

    def __init__(self, filename: str, site_meta: dict):
        self.filename = filename
        self.site_meta = site_meta
        self.raw_data: list = []
        self.data: dict = {}
        self.date: List[str] = []
        self.instrument = instruments.RADIOMETRICS

    def read_raw_data(self):
        """Reads radiometrics raw data."""
        with open(self.filename, mode="r", encoding="utf8") as infile:
            reader = csv.reader(infile)
            for x in reader:
                self.raw_data.append(x)
        self.raw_data = self.raw_data[1:]  # First row is header

    def read_lwp(self):
        """Reads LWP values."""
        self.data["lwp"] = np.array([row[9] for row in self.raw_data], dtype=float) * 1000  # g / m2

    def read_timestamps(self):
        """Reads timestamps."""
        fraction_hour = []
        time = [row[1].split()[1] for row in self.raw_data]
        for t in time:
            hour, minute, sec = t.split(":")
            fraction_hour.append(int(hour) + int(minute) / 60 + int(sec) / 3600)
        self.data["time"] = np.array(fraction_hour)

    def screen_time(self, expected_date: str = None):
        """Screens timestamps."""
        dates = [row[1].split()[0] for row in self.raw_data]
        if expected_date is None:
            self.date = self._convert_date(dates[0])
            return
        date_components = expected_date.split("-")
        self.date = date_components
        valid_ind = []
        valid_timestamps = []
        for ind, (d, timestamp) in enumerate(zip(dates, self.data["time"])):
            if self._convert_date(d) == date_components and timestamp not in valid_timestamps:
                valid_ind.append(ind)
                valid_timestamps.append(timestamp)
        if len(valid_ind) == 0:
            raise ValidTimeStampError
        for key, array in self.data.items():
            self.data[key] = array[valid_ind]

    def data_to_cloudnet_arrays(self):
        """Converts arrays to CloudnetArrays."""
        for key, array in self.data.items():
            self.data[key] = CloudnetArray(array, key)

    def add_meta(self):
        """Adds some metadata."""
        valid_keys = ("latitude", "longitude", "altitude")
        for key, value in self.site_meta.items():
            key = key.lower()
            if key in valid_keys:
                self.data[key] = CloudnetArray(float(value), key)

    @staticmethod
    def _convert_date(date: str) -> list:
        month, day, year = date.split("/")
        year = f"20{year}"
        return [year, month, day]
