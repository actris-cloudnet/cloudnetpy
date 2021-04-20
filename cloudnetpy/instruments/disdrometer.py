"""Module for reading / converting disdrometer data."""
from typing import Optional, Union
import numpy as np
import numpy.ma as ma
from cloudnetpy import output
from cloudnetpy.metadata import MetaData
from cloudnetpy.instruments.vaisala import values_to_dict
from cloudnetpy import CloudnetArray, utils


PARSIVEL = 'Parsivel'
THIES = 'Thies-LNM'


def disdrometer2nc(disdrometer_file: str,
                   output_file: str,
                   instrument: str,
                   site_meta: dict,
                   keep_uuid: Optional[bool] = False,
                   uuid: Optional[str] = None,
                   date: Optional[str] = None) -> str:
    """Converts disdrometer data into Cloudnet Level 1b netCDF file.

    Args:
        disdrometer_file: Filename of disdrometer .log file.
        output_file: Output filename.
        instrument: Disdrometer model. One of: parsivel|thies-lnm.
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
    if instrument == 'parsivel':
        disdrometer = Parsivel(disdrometer_file, site_meta)
    elif instrument == 'thies-lnm':
        disdrometer = Thies(disdrometer_file, site_meta)
    else:
        raise ValueError(f'Unknown disdrometer: {instrument}')
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
        self._file_contents, self._spectra = self._read_file()

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
                        print('Warning: skipping non-standard Parsivel data')
                        continue
                    scalars.append(values[:18])
                    vectors.append(values[18:18+64])  # not yet used
                    spectra.append(values[18+64:])
                else:
                    values = row.split(';')
                    scalars.append(values[:79])
                    spectra.append(values[79:-2])
        return scalars, spectra

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
        self.data['spectrum'] = CloudnetArray(np.array(self._spectra), 'spectrum')

    def _init_date(self) -> list:
        first_date = self._file_contents[0][3]
        first_date = _format_thies_date(first_date)
        return first_date.split('-')


class Parsivel(Disdrometer):
    def __init__(self, filename: str, site_meta: dict):
        super().__init__(filename, site_meta, PARSIVEL)
        self.date = self._init_date()
        self._create_velocity_vectors()
        self._create_diameter_vectors()

    def init_data(self):
        column_and_key = [
            (0, '_time'),
            (1, 'rain_rate'),  # liquid
            (2, 'rain_accum'),
            (4, 'radar_reflectivity'),
            (5, 'visibility'),
            (7, 'signal_amplitude'),
            (8, 'n_particles'),
            (9, 'sensor_temperature'),
            (12, 'heating_current'),
            (13, 'sensor_voltage'),
        ]
        self._append_data(column_and_key)
        self.data['spectrum'] = self._read_spectrum()

    def _read_spectrum(self) -> CloudnetArray:
        n_velocity = 32
        n_diameter = 32
        array = ma.masked_all((len(self._file_contents), n_velocity, n_diameter))
        for time_ind, row in enumerate(self._spectra):
            values = ma.masked_all((1024, ))
            for spec_ind, value in enumerate(row):
                try:
                    value = int(value)
                    if value != 0:
                        values[spec_ind] = value
                except ValueError:
                    pass
            array[time_ind, :, :] = np.reshape(values, (32, 32))
        return CloudnetArray(array, 'spectrum', dimensions=('time', 'velocity', 'diameter'))

    def _init_date(self) -> list:
        timestamp = self._file_contents[0][0]
        return _parse_parsivel_timestamp(timestamp)[:3]

    def _create_velocity_vectors(self):
        spreads = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2]
        start = 0.05
        lower, mid, upper = self._create_vectors(spreads, start)
        self.data['lower_velocity'] = CloudnetArray(lower, 'lower_velocity',
                                                    dimensions=('velocity',))
        self.data['upper_velocity'] = CloudnetArray(upper, 'upper_velocity',
                                                    dimensions=('velocity',))

    def _create_diameter_vectors(self):
        spreads = [0.125, 0.25, 0.5, 1, 2, 3]
        start = 0.062
        lower, mid, upper = self._create_vectors(spreads, start)
        self.data['lower_diameter'] = CloudnetArray(lower, 'lower_diameter',
                                                    dimensions=('diameter',))
        self.data['upper_diameter'] = CloudnetArray(upper, 'upper_diameter',
                                                    dimensions=('diameter',))

    @staticmethod
    def _create_vectors(spreads: list, start: float) -> tuple:
        n_values = [10, 5, 5, 5, 5, 2]
        mid_value, lower_limit, upper_limit = [], [], []
        for spread, n in zip(spreads, n_values):
            mid = np.linspace(start, start + (n-1)*spread, n)
            velocity = np.append(mid_value, mid)
            lower_limit = np.append(lower_limit, mid - spread/2)
            upper_limit = np.append(upper_limit, mid + spread/2)
            start = velocity[-1] + spread*1.5
        return lower_limit, mid_value, upper_limit


def save_disdrometer(disdrometer: Union[Parsivel, Thies],
                     output_file: str,
                     keep_uuid: bool,
                     uuid: Union[str, None]) -> str:
    """Saves disdrometer file."""

    dims = {'time': len(disdrometer.data['time'][:])}
    if disdrometer.source == PARSIVEL:
        dims['diameter'] = disdrometer.data['lower_diameter'][:].shape[0]
        dims['velocity'] = disdrometer.data['lower_velocity'][:].shape[0]
    else:
        dims['spectrum'] = disdrometer.data['spectrum'][:].shape[1]
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


ATTRIBUTES = {
    'precipitation_intensity': MetaData(
        long_name='Precipitation intensity',
        units='mm/h',
        comment='Rain droplets only'
    ),
    'radar_reflectivity': MetaData(
        long_name='Radar reflectivity',
        units='dBZ'
    ),
    'visibility': MetaData(
        long_name='Meteorological optical range visibility',
        units='m'
    ),
    'number_of_particles': MetaData(
        long_name='Number of detected particles',
    ),
    'kinetic_energy': MetaData(
        long_name='Kinetic energy',
    ),
    'signal_amplitude': MetaData(
        long_name='Signal amplitude',
    ),
    'sensor_temperature': MetaData(
        long_name='Sensor temperature',
        units='C'
    ),
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
    'spectrum': MetaData(
        long_name='Spectrum'
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
}
