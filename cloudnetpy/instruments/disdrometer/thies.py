from cloudnetpy import output
from cloudnetpy.exceptions import DisdrometerDataError
from cloudnetpy.instruments import instruments

from .common import ATTRIBUTES, THIES, Disdrometer, _format_thies_date


def thies2nc(
    disdrometer_file: str,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | None = None,
) -> str:
    """Converts Thies-LNM disdrometer data into Cloudnet Level 1b netCDF file.

    Args:
    ----
        disdrometer_file: Filename of disdrometer .log file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
    -------
        UUID of the generated file.

    Raises:
    ------
        DisdrometerDataError: Timestamps do not match the expected date, or unable
            to read the disdrometer file.

    Examples:
    --------
        >>> from cloudnetpy.instruments import thies2nc
        >>> site_meta = {'name': 'Lindenberg', 'altitude': 104, 'latitude': 52.2,
        'longitude': 14.1}
        >>> uuid = thies2nc('thies-lnm.log', 'thies-lnm.nc', site_meta)

    """
    try:
        disdrometer = Thies(disdrometer_file, site_meta)
    except (ValueError, IndexError) as err:
        msg = "Unable to read disdrometer file"
        raise DisdrometerDataError(msg) from err
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
    return output.save_level1b(disdrometer, output_file, uuid)


class Thies(Disdrometer):
    def __init__(self, filename: str, site_meta: dict):
        super().__init__(filename, site_meta, THIES)
        self.n_velocity = 20
        self.n_diameter = 22
        self.date = self._init_date()
        self._create_velocity_vectors()
        self._create_diameter_vectors()
        self.instrument = instruments.THIES

    def init_data(self) -> None:
        """According to
        https://www.biral.com/wp-content/uploads/2015/01/5.4110.xx_.xxx_.pdf
        """
        column_and_key = [
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
        self._append_data(column_and_key)
        self._append_spectra()

    def _init_date(self) -> list:
        first_date = self._file_data["scalars"][0][3]
        first_date = _format_thies_date(first_date)
        return first_date.split("-")

    def _create_velocity_vectors(self) -> None:
        n_values = [5, 6, 7, 1, 1]
        spreads = [0.2, 0.4, 0.8, 1, 10]
        self.store_vectors(self.data, n_values, spreads, "velocity")

    def _create_diameter_vectors(self) -> None:
        n_values = [3, 6, 13]
        spreads = [0.125, 0.25, 0.5]
        self.store_vectors(self.data, n_values, spreads, "diameter", start=0.125)
