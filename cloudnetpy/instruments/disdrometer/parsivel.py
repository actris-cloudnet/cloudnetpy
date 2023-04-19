import numpy as np
from numpy import ma

from cloudnetpy import output
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.exceptions import DisdrometerDataError
from cloudnetpy.instruments import instruments

from .common import (
    ATTRIBUTES,
    PARSIVEL,
    Disdrometer,
    _parse_int,
    _parse_parsivel_timestamp,
)


def parsivel2nc(
    disdrometer_file: str,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | None = None,
) -> str:
    """Converts OTT Parsivel-2 disdrometer data into Cloudnet Level 1b netCDF
    file.

    Args:
        disdrometer_file: Filename of disdrometer .log file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

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
    try:
        disdrometer = Parsivel(disdrometer_file, site_meta)
    except ValueError as err:
        raise DisdrometerDataError("Can not read disdrometer file") from err
    if date is not None:
        disdrometer.validate_date(date)
    disdrometer.init_data()
    if date is not None:
        disdrometer.sort_timestamps()
        disdrometer.remove_duplicate_timestamps()
    disdrometer.add_meta()
    disdrometer.convert_units()
    attributes = output.add_time_attribute(ATTRIBUTES, disdrometer.date)
    output.update_attributes(disdrometer.data, attributes)
    uuid = output.save_level1b(disdrometer, output_file, uuid)
    return uuid


class Parsivel(Disdrometer):
    def __init__(self, filename: str, site_meta: dict):
        super().__init__(filename, site_meta, PARSIVEL)
        self.n_velocity = 32
        self.n_diameter = 32
        self.date = self._init_date()
        self._create_velocity_vectors()
        self._create_diameter_vectors()
        self.instrument = instruments.PARSIVEL2

    def init_data(self):
        """
        Note:
            This is a custom format submitted by Juelich, Norunda and Ny-Alesund
            to Cloudnet data portal. It does not follow the order in the Parsivel2
            manual https://www.fondriest.com/pdf/ott_parsivel2_manual.pdf
        """
        column_and_key = [
            (0, "_time"),
            (1, "rainfall_rate"),
            (2, "_rain_accum"),
            (3, "synop_WaWa"),
            (4, "radar_reflectivity"),
            (5, "visibility"),
            (6, "interval"),
            (7, "sig_laser"),
            (8, "n_particles"),
            (9, "T_sensor"),
            (10, "_sensor_id"),  # to global attributes
            (12, "I_heating"),
            (13, "V_power_supply"),
            (14, "state_sensor"),
            (15, "_station_name"),
            (16, "_rain_amount_absolute"),
            (17, "error_code"),
        ]
        self._append_data(column_and_key)
        self._append_vector_data()
        self._append_spectra()

    def _append_vector_data(self):
        keys = ("number_concentration", "fall_velocity")
        data = {
            key: ma.masked_all((len(self._file_data["vectors"]), self.n_diameter))
            for key in keys
        }
        for time_ind, row in enumerate(self._file_data["vectors"]):
            values = _parse_int(row)
            for key, array in zip(keys, np.split(values, 2)):
                data[key][time_ind, :] = array
        for key in keys:
            self.data[key] = CloudnetArray(
                data[key], key, dimensions=("time", "diameter")
            )

    def _init_date(self) -> list:
        timestamp = self._file_data["scalars"][0][0]
        return _parse_parsivel_timestamp(timestamp)[:3]

    def _create_velocity_vectors(self):
        n_values = [10, 5, 5, 5, 5, 2]
        spreads = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        self._store_vectors(n_values, spreads, "velocity")

    def _create_diameter_vectors(self):
        n_values = [10, 5, 5, 5, 5, 2]
        spreads = [0.125, 0.25, 0.5, 1, 2, 3]
        self._store_vectors(n_values, spreads, "diameter")
