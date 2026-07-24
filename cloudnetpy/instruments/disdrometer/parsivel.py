import datetime
from collections import defaultdict
from collections.abc import Iterable, Sequence
from os import PathLike
from uuid import UUID

import numpy as np

from cloudnetpy.disdronator.parsivel import read_parsivel, read_parsivel_l1
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.disdrometer.rd80 import Disdro, _process_disdrometer

from .common import ATTRIBUTES


def parsivel2nc(
    disdrometer_file: str | PathLike | Iterable[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
    telegram: Sequence[int | None] | None = None,
    timestamps: Sequence[datetime.datetime] | None = None,
) -> UUID:
    """Converts OTT Parsivel-2 disdrometer data into Cloudnet Level 1b netCDF file.

    Args:
        disdrometer_file: Filename of disdrometer file or list of filenames.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.
        telegram: List of measured value numbers as specified in section 11.2 of
            the instrument's operating instructions. Unknown values are indicated
            with None. Telegram is required if the input file doesn't contain a
            header.
        timestamps: Specify list of timestamps if they are missing in the input file.

    Returns:
        UUID of the generated file.

    Raises:
        DisdrometerDataError: Timestamps do not match the expected date, or unable
            to read the disdrometer file.

    Examples:
        >>> from cloudnetpy.instruments import parsivel2nc
        >>> site_meta = {'name': 'Lindenberg', 'altitude': 104, 'latitude': 52.2,
        'longitude': 14.1}
        >>> uuid = parsivel2nc('parsivel.log', 'parsivel.nc', site_meta)

    """
    return _process_disdrometer(
        Parsivel, ATTRIBUTES, disdrometer_file, output_file, site_meta, uuid, date
    )


class Parsivel(Disdro):
    def __init__(
        self,
        filenames: Iterable[str | PathLike],
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__(filenames, site_meta, expected_date)
        self.serial_number = None
        self.instrument = instruments.PARSIVEL2

    def _read_data(self, filenames: Iterable[str | PathLike]) -> None:
        times = []
        data = defaultdict(list)
        for filename in filenames:
            file_time, file_data = read_parsivel(filename)
            times.append(file_time)
            for key, value in file_data.items():
                data[key].append(value)
        raw_time = np.concatenate(times)
        raw_data = {key: np.concatenate(value) for key, value in data.items()}
        return read_parsivel_l1(raw_time, raw_data)
