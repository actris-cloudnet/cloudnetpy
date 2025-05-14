import csv
import datetime
from os import PathLike

import numpy as np

from cloudnetpy import output
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CSVFile


def rain_e_h32nc(
    input_file: str | PathLike,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | datetime.date | None = None,
):
    """Converts rain_e_h3 rain-gauge into Cloudnet Level 1b netCDF file.

    Args:
        input_file: Filename of rain_e_h3 CSV file.
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
    rain = RainEH3(site_meta)
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    rain.parse_input_file(input_file, date)
    rain.add_data()
    rain.add_date()
    rain.convert_units()
    rain.normalize_cumulative_amount("rainfall_amount")
    rain.add_site_geolocation()
    rain.sort_timestamps()
    rain.remove_duplicate_timestamps()
    attributes = output.add_time_attribute({}, rain.date)
    output.update_attributes(rain.data, attributes)
    return output.save_level1b(rain, output_file, uuid)


class RainEH3(CSVFile):
    def __init__(self, site_meta: dict):
        super().__init__(site_meta)
        self.instrument = instruments.RAIN_E_H3
        self._data = {
            "time": [],
            "rainfall_rate": [],
            "rainfall_amount": [],
        }

    def parse_input_file(
        self, filepath: str | PathLike, date: datetime.date | None = None
    ) -> None:
        with open(filepath, encoding="latin1") as f:
            data = list(csv.reader(f, delimiter=";"))
        n_values = np.median([len(row) for row in data]).astype(int)

        if n_values == 22:
            self._read_talker_protocol_22_columns(data, date)
        elif n_values == 16:
            self._read_talker_protocol_16_columns(data, date)
        else:
            msg = "Only talker protocol with 16 or 22 columns is supported."
            raise NotImplementedError(msg)

    def _read_talker_protocol_16_columns(
        self, data: list, date: datetime.date | None = None
    ) -> None:
        """Old Lindenberg data format.

        0  date  DD.MM.YYYY
        1  time
        2  precipitation intensity in mm/h
        3  precipitation accumulation in mm
        4  housing contact
        5  top temperature
        6  bottom temperature
        7  heater status
        8  error code
        9  system status
        10 talker interval in seconds
        11 operating hours
        12 device type
        13 user data storage 1
        14 user data storage 2
        15 user data storage 3

        """
        for row in data:
            if len(row) != 16:
                continue
            try:
                dt = datetime.datetime.strptime(
                    f"{row[0]} {row[1]}", "%d.%m.%Y %H:%M:%S"
                )
            except ValueError:
                continue
            if date and date != dt.date():
                continue
            self._data["time"].append(dt)
            self._data["rainfall_rate"].append(float(row[2]))
            self._data["rainfall_amount"].append(float(row[3]))
        if not self._data["time"]:
            raise ValidTimeStampError

    def _read_talker_protocol_22_columns(
        self, data: list, date: datetime.date | None = None
    ) -> None:
        """Columns according to header in Lindenberg data.

        0  datetime utc
        1  date
        2  time
        3  precipitation intensity in mm/h
        4  precipitation accumulation in mm
        5  housing contact
        6  top temperature
        7  bottom temperature
        8  heater status
        9  error code
        10 system status
        11 talker interval in seconds
        12 operating hours
        13 device type
        14 user data storage 1
        15 user data storage 2
        16 user data storage 3
        17 user data storage 4
        18 serial number
        19 hardware version
        20 firmware version
        21 external temperature * checksum

        """
        for row in data:
            if len(row) != 22:
                continue
            try:
                dt = datetime.datetime.strptime(f"{row[0]}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            if date and date != dt.date():
                continue
            self._data["time"].append(dt)
            self._data["rainfall_rate"].append(float(row[3]))
            self._data["rainfall_amount"].append(float(row[4]))
            self.serial_number = row[18]
        if not self._data["time"]:
            raise ValidTimeStampError

    def convert_units(self) -> None:
        rainfall_rate = self.data["rainfall_rate"][:]
        self.data["rainfall_rate"].data = rainfall_rate / 3600 / 1000  # mm/h -> m/s
        self.data["rainfall_amount"].data = (
            self.data["rainfall_amount"][:] / 1000
        )  # mm -> m
