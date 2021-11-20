"""Module for reading / converting disdrometer data."""
from typing import Optional, Union
import numpy as np
import numpy.ma as ma
from cloudnetpy import output
from cloudnetpy.metadata import MetaData
from cloudnetpy.instruments.vaisala import values_to_dict
from cloudnetpy import CloudnetArray, utils
from cloudnetpy.exceptions import DisdrometerDataError


PARSIVEL = 'OTT Parsivel-2'
THIES = 'Thies-LNM'


def disdrometer2nc(disdrometer_file: str,
                   output_file: str,
                   site_meta: dict,
                   uuid: Optional[str] = None,
                   date: Optional[str] = None) -> str:
    """Converts disdrometer data into Cloudnet Level 1b netCDF file. Accepts measurements from
    OTT Parsivel-2 and Thies-LNM disdrometers.

    Args:
        disdrometer_file: Filename of disdrometer .log file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
        UUID of the generated file.

    Raises:
        DisdrometerDataError: Timestamps do not match the expected date, or unable to read
            the disdrometer file.

    Examples:
        >>> from cloudnetpy.instruments import disdrometer2nc
        >>> site_meta = {'name': 'Lindenberg', 'altitude': 104, 'latitude': 52.2, 'longitude': 14.1}
        >>> uuid = disdrometer2nc('thies-lnm.log', 'thies-lnm.nc', site_meta)

    """
    try:
        disdrometer = Parsivel(disdrometer_file, site_meta)
    except ValueError:
        try:
            disdrometer = Thies(disdrometer_file, site_meta)
        except (ValueError, IndexError):
            raise DisdrometerDataError('Can not read disdrometer file')
    if date is not None:
        disdrometer.validate_date(date)
    disdrometer.init_data()
    if date is not None:
        disdrometer.sort_time()
    disdrometer.add_meta()
    disdrometer.convert_units()
    attributes = output.add_time_attribute(ATTRIBUTES, disdrometer.date)
    output.update_attributes(disdrometer.data, attributes)
    return save_disdrometer(disdrometer, output_file, uuid)


class Disdrometer:
    def __init__(self, filename: str, site_meta: dict, source: str):
        self.filename = filename
        self.site_meta = site_meta
        self.source = source
        self.date = None
        self.sensor_id = None
        self.n_diameter = None
        self.n_velocity = None
        self.data = {}
        self._file_data = self._read_file()

    def convert_units(self):
        mm_to_m = 1e3
        mmh_to_ms = 3600 * mm_to_m
        c_to_k = 273.15
        self._convert_data(('rainfall_rate',), mmh_to_ms)
        self._convert_data(('diameter', 'diameter_spread', 'diameter_bnds'), mm_to_m)
        self._convert_data(('T_sensor',), c_to_k, method='add')
        self._convert_data(('V_sensor_supply',), 10)
        self._convert_data(('I_mean_laser',), 100)

    def add_meta(self):
        valid_keys = ('latitude', 'longitude', 'altitude')
        for key, value in self.site_meta.items():
            key = key.lower()
            if key in valid_keys:
                self.data[key] = CloudnetArray(value, key)

    def validate_date(self, expected_date: str) -> None:
        valid_ind = []
        for ind, row in enumerate(self._file_data['scalars']):
            if self.source == PARSIVEL:
                date = '-'.join(_parse_parsivel_timestamp(row[0])[:3])
            else:
                date = _format_thies_date(row[3])
            if date == expected_date:
                valid_ind.append(ind)
        if not valid_ind:
            raise ValueError('No measurements from expected date')
        for key, value in self._file_data.items():
            if value:
                self._file_data[key] = [self._file_data[key][ind] for ind in valid_ind]
        self.date = expected_date.split('-')

    def sort_time(self) -> None:
        time = self.data['time'][:]
        ind = time.argsort()
        for _, data in self.data.items():
            if data.data.shape[0] == len(time):
                data.data[:] = data.data[ind]

    def _read_file(self) -> dict:
        data = {
            'scalars': [],
            'vectors': [],
            'spectra': []
        }
        with open(self.filename, encoding="utf8", errors="ignore") as file:
            for row in file:
                if row == '\n':
                    continue
                if self.source == PARSIVEL:
                    values = row.split(';')
                    if '\n' in values:
                        values.remove('\n')
                    if len(values) != 1106:
                        continue
                    data['scalars'].append(values[:18])
                    data['vectors'].append(values[18:18+64])
                    data['spectra'].append(values[18+64:])
                else:
                    values = row.split(';')
                    data['scalars'].append(values[:79])
                    data['spectra'].append(values[79:-2])
        if len(data['scalars']) == 0:
            raise ValueError
        return data

    def _append_data(self, column_and_key: list) -> None:
        indices, keys = zip(*column_and_key)
        data = self._parse_useful_data(indices)
        data_dict = values_to_dict(keys, data)
        for key in keys:
            if key.startswith('_'):
                continue
            float_array = np.array([float(value) for value in data_dict[key]])
            self.data[key] = CloudnetArray(float_array, key)
        self.data['time'] = self._convert_time(data_dict)
        if '_sensor_id' in data_dict:
            self.sensor_id = data_dict['_sensor_id'][0]

    def _parse_useful_data(self, indices: list) -> list:
        data = []
        for row in self._file_data['scalars']:
            useful_data = [row[ind] for ind in indices]
            data.append(useful_data)
        return data

    def _convert_time(self, data: dict) -> CloudnetArray:
        seconds = []
        for timestamp in data['_time']:
            if self.source == PARSIVEL:
                _, _, _, hour, minute, sec = _parse_parsivel_timestamp(timestamp)
            else:
                hour, minute, sec = timestamp.split(':')
            seconds.append(int(hour)*3600 + int(minute)*60 + int(sec))
        return CloudnetArray(utils.seconds2hours(np.array(seconds)), 'time')

    def _convert_data(self, keys: tuple, value: float, method: Optional[str] = 'divide'):
        for key in keys:
            if key in self.data:
                if method == 'divide':
                    self.data[key].data /= value
                elif method == 'add':
                    self.data[key].data += value
                else:
                    raise ValueError

    def _append_spectra(self):
        array = ma.masked_all((len(self._file_data['scalars']), self.n_diameter, self.n_velocity))
        for time_ind, row in enumerate(self._file_data['spectra']):
            values = _parse_int(row)
            array[time_ind, :, :] = np.reshape(values, (self.n_diameter, self.n_velocity))
        self.data['data_raw'] = CloudnetArray(array, 'data_raw', dimensions=('time', 'diameter',
                                                                             'velocity'))

    def _store_vectors(self, n_values: list, spreads: list, name: str, start: Optional[float] = 0):
        mid, bounds, spread = self._create_vectors(n_values, spreads, start)
        self.data[name] = CloudnetArray(mid, name, dimensions=(name,))
        key = f'{name}_spread'
        self.data[key] = CloudnetArray(spread, key, dimensions=(name,))
        key = f'{name}_bnds'
        self.data[key] = CloudnetArray(bounds, key, dimensions=(name, 'nv'))

    @staticmethod
    def _create_vectors(n_values: list, spreads: list, start: float) -> tuple:
        mid_value, lower_limit, upper_limit = [], [], []
        for spread, n in zip(spreads, n_values):
            lower = np.linspace(start, start + (n-1)*spread, n)
            upper = lower + spread
            lower_limit = np.append(lower_limit, lower)
            upper_limit = np.append(upper_limit, upper)
            mid_value = np.append(mid_value, (lower + upper) / 2)
            start = upper[-1]
        bounds = np.stack((lower_limit, upper_limit)).T
        spread = bounds[:, 1] - bounds[:, 0]
        return mid_value, bounds, spread


class Parsivel(Disdrometer):
    def __init__(self, filename: str, site_meta: dict):
        super().__init__(filename, site_meta, PARSIVEL)
        self.n_velocity = 32
        self.n_diameter = 32
        self.date = self._init_date()
        self._create_velocity_vectors()
        self._create_diameter_vectors()

    def init_data(self):
        """
        Note:
            This is a custom format submitted by Juelich, Norunda and Ny-Alesund to Cloudnet
            data portal. It does not follow the order in the Parsivel2 manual
            https://www.fondriest.com/pdf/ott_parsivel2_manual.pdf
        """
        column_and_key = [
            (0, '_time'),
            (1, 'rainfall_rate'),
            (2, '_rain_accum'),
            (3, 'synop_WaWa'),
            (4, 'radar_reflectivity'),
            (5, 'visibility'),
            (6, 'interval'),
            (7, 'sig_laser'),
            (8, 'n_particles'),
            (9, 'T_sensor'),
            (10, '_sensor_id'),  # to global attributes
            (12, 'I_heating'),
            (13, 'V_sensor'),
            (14, 'state_sensor'),
            (15, '_station_name'),
            (16, '_rain_amount_absolute'),
            (17, 'error_code')
        ]
        self._append_data(column_and_key)
        self._append_vector_data()
        self._append_spectra()

    def _append_vector_data(self):
        keys = ('number_concentration', 'fall_velocity')
        data = {key: ma.masked_all((len(self._file_data['vectors']), self.n_diameter))
                for key in keys}
        for time_ind, row in enumerate(self._file_data['vectors']):
            values = _parse_int(row)
            for key, array in zip(keys, np.split(values, 2)):
                data[key][time_ind, :] = array
        for key in keys:
            self.data[key] = CloudnetArray(data[key], key, dimensions=('time', 'diameter'))

    def _init_date(self) -> list:
        timestamp = self._file_data['scalars'][0][0]
        return _parse_parsivel_timestamp(timestamp)[:3]

    def _create_velocity_vectors(self):
        n_values = [10, 5, 5, 5, 5, 2]
        spreads = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        self._store_vectors(n_values, spreads, 'velocity')

    def _create_diameter_vectors(self):
        n_values = [10, 5, 5, 5, 5, 2]
        spreads = [0.125, 0.25, 0.5, 1, 2, 3]
        self._store_vectors(n_values, spreads, 'diameter')


class Thies(Disdrometer):
    def __init__(self, filename: str, site_meta: dict):
        super().__init__(filename, site_meta, THIES)
        self.n_velocity = 20
        self.n_diameter = 22
        self.date = self._init_date()
        self._create_velocity_vectors()
        self._create_diameter_vectors()

    def init_data(self):
        """According to https://www.biral.com/wp-content/uploads/2015/01/5.4110.xx_.xxx_.pdf"""
        column_and_key = [
            (1, '_serial_number'),
            (2, '_software_version'),
            (3, '_date'),
            (4, '_time'),
            (5, '_synop_5min_ww'),
            (6, '_synop_5min_WaWa'),
            (7, '_metar_5min_4678'),
            (8, '_rainfall_rate_5min'),
            (9, 'synop_WW'),  # 1min
            (10, 'synop_WaWa'),  # 1min
            (11, '_metar_1_min_4678'),
            (12, '_rainfall_rate_1min_total'),
            (13, 'rainfall_rate'),                # liquid, mm h-1
            (14, '_rainfall_rate_1min_solid'),
            (15, '_precipition_amount'),  # mm
            (16, 'visibility'),
            (17, 'radar_reflectivity'),
            (18, 'measurement_quality'),
            (19, 'maximum_hail_diameter'),
            (20, 'status_laser'),
            (21, 'static_signal'),
            (22, 'status_T_laser_analogue'),
            (23, 'status_T_laser_digital'),
            (24, 'status_I_laser_analogue'),
            (25, 'status_I_laser_digital'),
            (26, 'status_sensor_supply'),
            (27, 'status_laser_heating'),
            (28, 'status_receiver_heating'),
            (29, 'status_temperature_sensor'),
            (30, 'status_heating_supply'),
            (31, 'status_heating_housing'),
            (32, 'status_heating_heads'),
            (33, 'status_heating_carriers'),
            (34, 'status_laser_power'),
            (35, '_status_reserve'),
            (36, 'T_interior'),
            (37, 'T_laser_driver'),   # 0-80 C
            (38, 'I_mean_laser'),
            (39, 'V_control'),  # mV 4005-4015
            (40, 'V_optical_output'),  # mV 2300-6500
            (41, 'V_sensor_supply'),  # 1/10V
            (42, 'I_heating_laser_head'),  # mA
            (43, 'I_heating_receiver_head'),  # mA
            (44, 'T_ambient'),  # C
            (45, '_V_heating_supply'),
            (46, '_I_housing'),
            (47, '_I_heating_heads'),
            (48, '_I_heating_carriers'),
            (49, 'n_particles'),
        ]
        self._append_data(column_and_key)
        self._append_spectra()

    def _init_date(self) -> list:
        first_date = self._file_data['scalars'][0][3]
        first_date = _format_thies_date(first_date)
        return first_date.split('-')

    def _create_velocity_vectors(self):
        n_values = [5, 6, 7, 1, 1]
        spreads = [0.2, 0.4, 0.8, 1, 10]
        self._store_vectors(n_values, spreads, 'velocity')

    def _create_diameter_vectors(self):
        n_values = [3, 6, 13]
        spreads = [0.125, 0.25, 0.5]
        self._store_vectors(n_values, spreads, 'diameter', start=0.125)


def save_disdrometer(disdrometer: Union[Parsivel, Thies],
                     output_file: str,
                     uuid: Union[str, None]) -> str:
    """Saves disdrometer file."""
    dims = {
        'time': len(disdrometer.data['time'][:]),
        'diameter': disdrometer.n_diameter,
        'velocity': disdrometer.n_velocity,
        'nv': 2
    }
    file_type = 'disdrometer'
    rootgrp = output.init_file(output_file, dims, disdrometer.data, uuid)
    file_uuid = rootgrp.file_uuid
    rootgrp.cloudnet_file_type = file_type
    rootgrp.title = f"{file_type.capitalize()} file from {disdrometer.site_meta['name']}"
    rootgrp.year, rootgrp.month, rootgrp.day = disdrometer.date
    rootgrp.location = disdrometer.site_meta['name']
    rootgrp.history = f"{utils.get_time()} - {file_type} file created"
    rootgrp.source = disdrometer.source
    if disdrometer.sensor_id is not None:
        rootgrp.sensor_id = disdrometer.sensor_id
    rootgrp.references = output.get_references()
    rootgrp.close()
    return file_uuid


def _parse_parsivel_timestamp(timestamp: str) -> list:
    year = timestamp[:4]
    month = timestamp[4:6]
    day = timestamp[6:8]
    hour = timestamp[8:10]
    minute = timestamp[10:12]
    second = timestamp[12:14]
    return [year, month, day, hour, minute, second]


def _format_thies_date(date: str):
    day, month, year = date.split('.')
    year = f'20{year}'
    return f'{year}-{month.zfill(2)}-{day.zfill(2)}'


def _parse_int(row: np.ndarray) -> np.ndarray:
    values = ma.masked_all((len(row),))
    for ind, value in enumerate(row):
        try:
            value = int(value)
            if value != 0:
                values[ind] = value
        except ValueError:
            pass
    return values


ATTRIBUTES = {
    'velocity': MetaData(
        long_name='Center fall velocity of precipitation particles',
        units='m s-1',
        comment='Predefined velocity classes.'
    ),
    'velocity_spread': MetaData(
        long_name='Width of velocity interval',
        units='m s-1',
        comment='Bin size of each velocity interval.'
    ),
    'velocity_bnds': MetaData(
        long_name='Velocity bounds',
        units='m s-1',
        comment='Upper and lower bounds of velocity interval.'
    ),
    'diameter': MetaData(
        long_name='Center diameter of precipitation particles',
        units='m',
        comment='Predefined diameter classes.'
    ),
    'diameter_spread': MetaData(
        long_name='Width of diameter interval',
        units='m',
        comment='Bin size of each diameter interval.'
    ),
    'diameter_bnds': MetaData(
        long_name='Diameter bounds',
        units='m',
        comment='Upper and lower bounds of diameter interval.'
    ),
    'rainfall_rate': MetaData(
        long_name='Precipitation rate',
        units='m s-1',
    ),
    'synop_WaWa': MetaData(
        long_name='Synop code WaWa',
    ),
    'synop_WW': MetaData(
        long_name='Synop code WW',
    ),
    'radar_reflectivity': MetaData(
        long_name='Equivalent radar reflectivity factor',
        units='dBZ',
    ),
    'visibility': MetaData(
        long_name='Visibility range in precipitation after MOR',
        units='m',
    ),
    'interval': MetaData(
        long_name='Length of measurement interval',
    ),
    'sig_laser': MetaData(
        long_name='Signal amplitude of the laser',
    ),
    'n_particles': MetaData(
        long_name='Number of particles in time interval',
    ),
    'T_sensor': MetaData(
        long_name='Temperature in the sensor',
        units='K',
    ),
    'I_heating': MetaData(
        long_name='Heating current',
        units='A',
    ),
    'V_sensor': MetaData(
        long_name='Sensor voltage',
        units='V',
    ),
    'state_sensor': MetaData(
        long_name='State of the sensor',
        comment='0 = OK, 1 = Dirty, 2 = No measurement possible.'
    ),
    'error_code': MetaData(
        long_name='Error code',
    ),
    'number_concentration': MetaData(
        long_name='Number of particles per diameter class',
        units='log10(m-3 mm-1)',
    ),
    'fall_velocity': MetaData(
        long_name='Average velocity of each diameter class',
        units='m s-1',
    ),
    'data_raw': MetaData(
        long_name='Raw Data as a function of particle diameter and velocity.',
    ),
    'kinetic_energy': MetaData(
        long_name='Kinetic energy',
    ),
    # Thies-specific:
    'T_ambient': MetaData(
        long_name='Ambient temperature',
        units='C'
    ),
    'T_interior': MetaData(
        long_name='Interior temperature',
        units='C'
    ),
    'status_T_laser_analogue': MetaData(
        long_name='Status of laser temperature (analogue)',
        comment='0 = OK , 1 = Error'
    ),
    'status_T_laser_digital': MetaData(
        long_name='Status of laser temperature (digital)',
        comment='0 = OK , 1 = Error'
    ),
    'status_I_laser_analogue': MetaData(
        long_name='Status of laser current (analogue)',
        comment='0 = OK , 1 = Error'
    ),
    'status_I_laser_digital': MetaData(
        long_name='Status of laser current (digital)',
        comment='0 = OK , 1 = Error'
    ),
    'status_sensor_supply': MetaData(
        long_name='Status of sensor supply',
        comment='0 = OK , 1 = Error'
    ),
    'status_laser_heating': MetaData(
        long_name='Status of laser heating',
        comment='0 = OK , 1 = Error'
    ),
    'status_receiver_heating': MetaData(
        long_name='Status of receiver heating',
        comment='0 = OK , 1 = Error'
    ),
    'status_temperature_sensor': MetaData(
        long_name='Status of temperature sensor',
        comment='0 = OK , 1 = Error'
    ),
    'status_heating_supply': MetaData(
        long_name='Status of heating supply',
        comment='0 = OK , 1 = Error'
    ),
    'status_heating_housing': MetaData(
        long_name='Status of heating housing',
        comment='0 = OK , 1 = Error'
    ),
    'status_heating_heads': MetaData(
        long_name='Status of heating heads',
        comment='0 = OK , 1 = Error'
    ),
    'status_heating_carriers': MetaData(
        long_name='Status of heating carriers',
        comment='0 = OK , 1 = Error'
    ),
    'status_laser_power': MetaData(
        long_name='Status of laser power',
        comment='0 = OK , 1 = Error'
    ),
    'status_laser': MetaData(
        long_name='Status of laser',
        comment='0 = OK/on , 1 = Off'
    ),
    'measurement_quality': MetaData(
        long_name='Measurement quality',
        units='%'
    ),
    'maximum_hail_diameter': MetaData(
        long_name='Maximum hail diameter',
        units='mm'
    ),
    'static_signal': MetaData(
        long_name='Static signal',
        comment='0 = OK, 1 = ERROR'
    ),
    'T_laser_driver': MetaData(
        long_name='Temperature of laser driver',
        units='C'
    ),
    'I_mean_laser': MetaData(
        long_name='Mean value of laser current',
        units='mA'
    ),
    'V_control': MetaData(
        long_name='Control voltage',
        units='mV',
        comment='Reference value: 4010+-5'
    ),
    'V_optical_output': MetaData(
        long_name='Voltage of optical control output',
        units='mV',
    ),
    'V_sensor_supply': MetaData(
        long_name='Voltage of sensor supply',
        units='V',
    ),
    'I_heating_laser_head': MetaData(
        long_name='Laser head heating current',
        units='mA',
    ),
    'I_heating_receiver_head': MetaData(
        long_name='Receiver head heating current',
        units='mA',
    ),
}
