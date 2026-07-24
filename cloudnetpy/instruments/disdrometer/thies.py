import datetime
from collections import defaultdict
from collections.abc import Iterable
from os import PathLike
from uuid import UUID

import numpy as np

from cloudnetpy.disdronator.lpm import read_lpm, read_lpm_l1
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.disdrometer.rd80 import Disdro, _process_disdrometer

from .common import ATTRIBUTES

TELEGRAM4 = [
    (1, "_serial_number"),
    (2, "_software_version"),
    (3, "_date"),
    (4, "_time"),
    (5, "_synop_5min_ww"),
    (6, "_synop_5min_WaWa"),
    (7, "_metar_5min_4678"),
    (8, "_rainfall_rate_5min"),
    (9, "synop_WW"),  # 1min
    (10, "synop_WaWa"),  # 1min
    (11, "_metar_1_min_4678"),
    (12, "rainfall_rate_1min_total"),
    (13, "rainfall_rate"),  # liquid, mm h-1
    (14, "rainfall_rate_1min_solid"),
    (15, "_precipition_amount"),  # mm
    (16, "visibility"),
    (17, "radar_reflectivity"),
    (18, "measurement_quality"),
    (19, "maximum_hail_diameter"),
    (20, "status_laser"),
    (21, "static_signal"),
    (22, "status_T_laser_analogue"),
    (23, "status_T_laser_digital"),
    (24, "status_I_laser_analogue"),
    (25, "status_I_laser_digital"),
    (26, "status_sensor_supply"),
    (27, "status_laser_heating"),
    (28, "status_receiver_heating"),
    (29, "status_temperature_sensor"),
    (30, "status_heating_supply"),
    (31, "status_heating_housing"),
    (32, "status_heating_heads"),
    (33, "status_heating_carriers"),
    (34, "status_laser_power"),
    (35, "_status_reserve"),
    (36, "T_interior"),
    (37, "T_laser_driver"),  # 0-80 C
    (38, "I_mean_laser"),
    (39, "V_control"),  # mV 4005-4015
    (40, "V_optical_output"),  # mV 2300-6500
    (41, "V_sensor_supply"),  # 1/10V
    (42, "I_heating_laser_head"),  # mA
    (43, "I_heating_receiver_head"),  # mA
    (44, "T_ambient"),  # C
    (45, "_V_heating_supply"),
    (46, "_I_housing"),
    (47, "_I_heating_heads"),
    (48, "_I_heating_carriers"),
    (49, "n_particles"),
]


def thies2nc(
    disdrometer_file: str | PathLike | Iterable[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts Thies-LNM disdrometer data into Cloudnet Level 1b netCDF file.

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
        >>> from cloudnetpy.instruments import thies2nc
        >>> site_meta = {'name': 'Lindenberg', 'altitude': 104, 'latitude': 52.2,
        'longitude': 14.1}
        >>> uuid = thies2nc('thies-lnm.log', 'thies-lnm.nc', site_meta)

    """
    return _process_disdrometer(
        Thies, ATTRIBUTES, disdrometer_file, output_file, site_meta, uuid, date
    )


class Thies(Disdro):
    def __init__(
        self,
        filenames: Iterable[str | PathLike],
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__(filenames, site_meta, expected_date)
        self.serial_number = None
        self.instrument = instruments.THIES

    def _read_data(self, filenames: Iterable[str | PathLike]) -> None:
        times = []
        data = defaultdict(list)
        for filename in filenames:
            file_time, file_data = read_lpm(filename)
            times.append(file_time)
            for key, value in file_data.items():
                data[key].append(value)
        raw_time = np.concatenate(times)
        raw_data = {key: np.concatenate(value) for key, value in data.items()}
        return read_lpm_l1(raw_time, raw_data)
