"""Module for reading / converting disdrometer data."""
from typing import Optional, Union
import numpy as np
import numpy.ma as ma
from cloudnetpy import output
from cloudnetpy.metadata import MetaData
from cloudnetpy.instruments.vaisala import values_to_dict
from cloudnetpy import CloudnetArray, utils
import logging


PARSIVEL = 'Parsivel'
THIES = 'Thies-LNM'


def disdrometer2nc(disdrometer_file: str,
                   output_file: str,
                   site_meta: dict,
                   keep_uuid: Optional[bool] = False,
                   uuid: Optional[str] = None,
                   date: Optional[str] = None) -> str:
    """Converts disdrometer data into Cloudnet Level 1b netCDF file.

    Args:
        disdrometer_file: Filename of disdrometer .log file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key is `name`.
        keep_uuid: If True, keeps the UUID of the old file, if that exists. Default is False
            when new UUID is generated.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
        UUID of the generated file.

    Raises:
        ValueError: Timestamps do not match the expected date, or unknown disdrometer model.

    """
    try:
        disdrometer = Parsivel(disdrometer_file, site_meta)
    except ValueError:
        try:
            disdrometer = Thies(disdrometer_file, site_meta)
        except IndexError:
            raise ValueError('Can not read disdrometer file')
    if date is not None:
        disdrometer.validate_date(date)
    disdrometer.init_data()
    attributes = output.add_time_attribute(ATTRIBUTES, disdrometer.date)
    output.update_attributes(disdrometer.data, attributes)
    return save_disdrometer(disdrometer, output_file, keep_uuid, uuid)


class Disdrometer:
    def __init__(self, filename: str, site_meta: dict, source: str):
        self.filename = filename
        self.location = site_meta['name']
        self.data = {}
        self.source = source
        self.date = None
        self._file_contents, self._spectra, self._vectors = self._read_file()

    def validate_date(self, expected_date: str) -> None:
        valid_ind = []
        for ind, row in enumerate(self._file_contents):
            if self.source == PARSIVEL:
                date = '-'.join(_parse_parsivel_timestamp(row[0])[:3])
            else:
                date = _format_thies_date(row[3])
            if date == expected_date:
                valid_ind.append(ind)
        if not valid_ind:
            raise ValueError('No measurements from expected date')
        self._file_contents = [self._file_contents[ind] for ind in valid_ind]
        self._spectra = [self._spectra[ind] for ind in valid_ind]
        self.date = expected_date.split('-')

    def _read_file(self) -> tuple:
        scalars, vectors, spectra = [], [], []
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
                    scalars.append(values[:18])
                    vectors.append(values[18:18+64])
                    spectra.append(values[18+64:])
                else:
                    values = row.split(';')
                    scalars.append(values[:79])
                    spectra.append(values[79:-2])
        if len(scalars) == 0:
            raise ValueError
        return scalars, spectra, vectors

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

    def _parse_useful_data(self, indices: list) -> list:
        data = []
        for row in self._file_contents:
            useful_data = [row[ind] for ind in indices]
            data.append(useful_data)
        return data

    def _convert_time(self, data: dict) -> CloudnetArray:
        seconds = []
        for ind, timestamp in enumerate(data['_time']):
            if self.source == PARSIVEL:
                _, _, _, hour, minute, sec = _parse_parsivel_timestamp(timestamp)
            else:
                hour, minute, sec = timestamp.split(':')
            seconds.append(int(hour)*3600 + int(minute)*60 + int(sec))
        return CloudnetArray(utils.seconds2hours(seconds), 'time')


class Parsivel(Disdrometer):
    def __init__(self, filename: str, site_meta: dict):
        super().__init__(filename, site_meta, PARSIVEL)
        self.date = self._init_date()
        self._create_velocity_vectors()
        self._create_diameter_vectors()

    def init_data(self):
        column_and_key = [
            (0, '_time'),
            (1, 'rainfall_rate'),
            (2, '_rain_accum'),
            (3, 'synop_WaWa'),  # or synop_ww ?
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
        self._append_spectra()
        self._append_vector_data()

    def _append_spectra(self):
        n_velocity, n_diameter = 32, 32
        array = ma.masked_all((len(self._file_contents), n_velocity, n_diameter))
        for time_ind, row in enumerate(self._spectra):
            values = _parse_int(row)
            array[time_ind, :, :] = np.reshape(values, (n_velocity, n_diameter))
        self.data['data_raw'] = CloudnetArray(array, 'data_raw', dimensions=('time', 'velocity',
                                                                             'diameter'))

    def _append_vector_data(self):
        n_diameter = 32
        keys = ('number_concentration', 'fall_velocity')
        data = {key: ma.masked_all((len(self._vectors), n_diameter)) for key in keys}
        for time_ind, row in enumerate(self._vectors):
            values = _parse_int(row)
            for key, array in zip(keys, np.split(values, 2)):
                data[key][time_ind, :] = array
        for key in keys:
            self.data[key] = CloudnetArray(data[key], key, dimensions=('time', 'diameter'))

    def _init_date(self) -> list:
        timestamp = self._file_contents[0][0]
        return _parse_parsivel_timestamp(timestamp)[:3]

    def _create_velocity_vectors(self):
        spreads = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        start = 0.05
        self._store_vectors(spreads, start, 'velocity')

    def _create_diameter_vectors(self):
        spreads = [0.125, 0.25, 0.5, 1, 2, 3]
        start = 0.062
        self._store_vectors(spreads, start, 'diameter')

    def _store_vectors(self, spreads: list, start: float, name: str):
        mid, bounds, spread = self._create_vectors(spreads, start)
        self.data[name] = CloudnetArray(mid, name, dimensions=(name,))
        key = f'{name}_spread'
        self.data[key] = CloudnetArray(spread, key, dimensions=(name,))
        key = f'{name}_bnds'
        self.data[key] = CloudnetArray(bounds, key, dimensions=(name, 2))

    @staticmethod
    def _create_vectors(spreads: list, start: float) -> tuple:
        n_values = [10, 5, 5, 5, 5, 2]
        mid_value, lower_limit, upper_limit = [], [], []
        for spread, n in zip(spreads, n_values):
            mid = np.linspace(start, start + (n-1)*spread, n)
            velocity = np.append(mid_value, mid)
            lower_limit = np.append(lower_limit, mid - spread/2)
            upper_limit = np.append(upper_limit, mid + spread/2)
            mid_value = np.append(mid_value, mid)
            start = velocity[-1] + spread*1.5
        bounds = np.stack((lower_limit, upper_limit)).T
        spread = bounds[:, 1] - bounds[:, 0]
        return mid_value, bounds, spread


class Thies(Disdrometer):

    def __init__(self, filename: str, site_meta: dict):
        super().__init__(filename, site_meta, THIES)
        self.date = self._init_date()

    def init_data(self):
        column_and_key = [
            (4, '_time'),
            (13, 'rain_rate'),  # liquid
            (14, 'snow_intensity'),
            (16, 'visibility'),
            (17, 'radar_reflectivity'),
            (18, 'measurement_quality'),
            (19, 'maximum_hail_diameter'),
            (20, 'laser_status'),
            (21, 'static_signal'),
            (44, 'ambient_temperature'),
            (49, 'n_particles')
        ]
        self._append_data(column_and_key)
        self.data['data_raw'] = CloudnetArray(np.array(self._spectra), 'data_raw')

    def _init_date(self) -> list:
        first_date = self._file_contents[0][3]
        first_date = _format_thies_date(first_date)
        return first_date.split('-')


def save_disdrometer(disdrometer: Union[Parsivel, Thies],
                     output_file: str,
                     keep_uuid: bool,
                     uuid: Union[str, None]) -> str:
    """Saves disdrometer file."""

    dims = {'time': len(disdrometer.data['time'][:])}
    if disdrometer.source == PARSIVEL:
        dims['diameter'] = disdrometer.data['diameter'][:].shape[0]
        dims['velocity'] = disdrometer.data['velocity'][:].shape[0]
        dims['nv'] = 2
    else:
        dims['data_raw'] = disdrometer.data['data_raw'][:].shape[1]
    file_type = 'disdrometer'
    rootgrp = output.init_file(output_file, dims, disdrometer.data, keep_uuid, uuid)
    file_uuid = rootgrp.file_uuid
    output.add_file_type(rootgrp, file_type)
    rootgrp.title = f"{file_type.capitalize()} file from {disdrometer.location}"
    rootgrp.year, rootgrp.month, rootgrp.day = disdrometer.date
    rootgrp.location = disdrometer.location
    rootgrp.history = f"{utils.get_time()} - {file_type} file created"
    rootgrp.source = disdrometer.source
    output.add_references(rootgrp)
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
        comment='Predefined velocity classes. Note the variable bin size.'
    ),
    'velocity_spread': MetaData(
        long_name='Width of velocity interval',
        units='m s-1',
        comment='Bin size of each velocity interval.'
    ),
    'velocity_bnds': MetaData(
        long_name='',
        units='m s-1',
        comment='Upper and lower bounds of velocity interval.'
    ),
    'diameter': MetaData(
        long_name='Center diameter of precipitation particles',
        units='m',
        comment='Predefined diameter classes. Note the variable bin size.'
    ),
    'diameter_spread': MetaData(
        long_name='Width of diameter interval',
        units='m',
        comment='Bin size of each diameter interval.'
    ),
    'diameter_bnds': MetaData(
        long_name='',
        units='m',
        comment='Upper and lower bounds of diameter interval.'
    ),
    'rainfall_rate': MetaData(
        long_name='Precipitation rate',
        units='mm h-1',
        comment='Variable 01 - Rain intensity (32 bit) 0000.000.'
    ),
    'radar_reflectivity': MetaData(
        long_name='Equivalent radar reflectivity factor',
        units='dBZ',
        comment='Variable 07 - Radar reflectivity (32 bit).'
    ),
    'visibility': MetaData(
        long_name='Visibility range in precipitation after MOR',
        units='m',
        comment='Variable 08 - MOR visibility in the precipitation.'
    ),
    'interval': MetaData(
        long_name='Length of measurement interval',
        comment='Variable 09 - Sample interval between two data retrieval request.'
    ),
    'sig_laser': MetaData(
        long_name='Signal amplitude of the laser',
        comment='Variable 10 - Signal amplitude of the laser strip.'
    ),
    'n_particles': MetaData(
        long_name='Number of particles in time interval',
        comment='Variable 11 - Number of detected particles.'
    ),
    'T_sensor': MetaData(
        long_name='Temperature in the sensor',
        units='C',
        comment='Variable 12 - Temperature in the sensor.'
    ),
    'I_heating': MetaData(
        long_name='Heating current',
        units='A',
        comment='Variable 16 - Current through the heating system.'
    ),
    'V_sensor': MetaData(
        long_name='Sensor voltage',
        units='V',
        comment='Variable 17 - Power supply voltage in the sensor.'
    ),
    'state_sensor': MetaData(
        long_name='State of the sensor',
        comment='Variable 18 - Sensor status: 0 = Everything is okay, 1 = Dirty, '
                '2 = No measurement possible.'
    ),
    'error_code': MetaData(
        long_name='Error code',
        comment='Variable 25 - Error code.'
    ),
    'number_concentration': MetaData(
        long_name='Number of particles per diameter class',
        units='log10(m-3 mm-1)',
        comment='Variable 90 - Field N (d)'
    ),
    'fall_velocity': MetaData(
        long_name='Average velocity of each diameter class',
        units='m s-1',
        comment='Variable 91 - Field v (d)'
    ),
    'data_raw': MetaData(
        long_name='Raw Data as a function of particle diameter and velocity.',
        comment='Variable 93 - Raw data.'
    ),
    # Thies-specific:
    'ambient_temperature': MetaData(
        long_name='Ambient temperature',
        units='C'
    ),
    'heating_current': MetaData(
        long_name='Heating current',
        units='A'
    ),
    'sensor_voltage': MetaData(
        long_name='Sensor voltage',
        units='V'
    ),
    'snow_intensity': MetaData(
        long_name='Snow intensity',
        units='mm/h'
    ),
    'measurement_quality': MetaData(
        long_name='Measurement quality',
        units='%'
    ),
    'maximum_hail_diameter': MetaData(
        long_name='Maximum hail diameter',
        units='mm'
    ),
    'laser_status': MetaData(
        long_name='Laser status',
        comment='0 = ON, 1 = OFF'
    ),
    'static_signal': MetaData(
        long_name='Static signal',
        comment='0 = OK, 1 = ERROR'
    ),
    'kinetic_energy': MetaData(
        long_name='Kinetic energy',
    ),
    'precipitation_intensity': MetaData(
        long_name='Precipitation intensity',
        units='mm/h',
        comment='Rain droplets only'
    ),
}
