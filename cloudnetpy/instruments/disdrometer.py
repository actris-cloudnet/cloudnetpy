"""Module for reading / converting disdrometer data."""
from typing import Optional, Union
import numpy as np
from cloudnetpy import output
from cloudnetpy.metadata import MetaData
from cloudnetpy.instruments.vaisala import values_to_dict
from cloudnetpy import CloudnetArray, utils


def disdrometer2nc(disdrometer_file: str,
                   output_file: str,
                   site_meta: dict,
                   keep_uuid: Optional[bool] = False,
                   uuid: Optional[str] = None,
                   date: Optional[str] = None) -> str:
    """Converts disdrometer data into Cloudnet Level 1b netCDF file.

    Args:
        disdrometer_file: Filename of disdrometer .txt file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key is `name`.
        keep_uuid: If True, keeps the UUID of the old file, if that exists. Default is False
            when new UUID is generated.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
        UUID of the generated file.

    Raises:
        ValueError: Timestamps do not match the expected date.

    """
    parsivel = Parsivel(disdrometer_file, site_meta)
    if date is not None:
        parsivel.validate_date(date)
    parsivel.init_data()
    attributes = output.add_time_attribute(ATTRIBUTES, parsivel.date)
    output.update_attributes(parsivel.data, attributes)
    return save_disdrometer(parsivel, output_file, keep_uuid, uuid)


class Parsivel:
    def __init__(self, filename: str, site_meta: dict):
        self.filename = filename
        self._file_contents = self._read_parsivel()
        self.source = 'Parsivel'
        self.location = site_meta['name']
        self.date = self._get_date()
        self.data = {}

    def validate_date(self, expected_date: str):
        self._file_contents = [row for row in self._file_contents
                               if row[0].replace('/', '-') == expected_date]
        if not self._file_contents:
            raise ValueError
        self.date = expected_date.split('-')

    def init_data(self):
        keys = ('_date', '_time', 'precipitation_intensity', '_precipitation_since_start',
                'radar_reflectivity', 'mor_visibility', 'signal_amplitude',
                'number_of_particles', 'sensor_temperature', 'heating_current',
                'sensor_voltage', 'kinetic_energy', 'snow_intensity', '_weather_code_synop_wawa',
                '_weather_code_metar/speci', '_weather_code_nws', '_spectrum')
        data_dict = values_to_dict(keys, self._file_contents)
        for key in keys:
            if key.startswith('_'):
                continue
            data_as_float = np.array([float(value) for value in data_dict[key]])
            self.data[key] = CloudnetArray(data_as_float, key)
        self.data['time'] = self.convert_time(data_dict)

    @staticmethod
    def convert_time(data: dict) -> CloudnetArray:
        seconds = []
        for ind, timestamp in enumerate(data['_time']):
            hour, minute, sec = timestamp.split(':')
            seconds.append(int(hour)*3600 + int(minute)*60 + int(sec))
        return CloudnetArray(utils.seconds2hours(seconds), 'time')

    def _read_parsivel(self) -> list:
        with open(self.filename, encoding="utf8", errors="ignore") as file:
            file.readline()
            return [row.split(';') for row in file if row != '\n']

    def _get_date(self):
        return self._file_contents[0][0].split('/')


def save_disdrometer(disdrometer: Parsivel,
                     output_file: str,
                     keep_uuid: bool,
                     uuid: Union[str, None]) -> str:
    """Saves disdrometer file."""

    dims = {'time': len(disdrometer.data['time'][:])}
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


ATTRIBUTES = {
    'precipitation_intensity': MetaData(
        long_name='Precipitation intensity',
        units='mm/h'
    ),
    'radar_reflectivity': MetaData(
        long_name='Radar reflectivity',
        units='dBZ'
    ),
    'mor_visibility': MetaData(
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
}
