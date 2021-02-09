"""This module contains RPG Cloud Radar related functions."""
from typing import Union, Tuple
import numpy as np
from cloudnetpy import utils, output
from cloudnetpy.metadata import MetaData
from cloudnetpy.instruments import rpg


def hatpro2nc(path_to_lwp_files: str,
              output_file: str,
              site_meta: dict,
              keep_uuid: bool = False,
              uuid: Union[str, None] = None,
              date: Union[str, None] = None) -> Tuple[str, list]:
    """Converts RPG HATPRO LWP binary files into netCDF file.

    This function reads one day of RPG HATPRO binary files,
    concatenates the data and writes it into netCDF file.

    Args:
        path_to_lwp_files (str): Folder containing one day of RPG HATPRO files.
        output_file (str): Output file name.
        site_meta (dict): Dictionary containing information about the
            site. Required key value pairs are `altitude` (metres above mean
            sea level) and `name`.
        keep_uuid (bool, optional): If True, keeps the UUID of the old file,
            if that exists. Default is False when new UUID is generated.
        uuid (str, optional): Set specific UUID for the file.
        date (str, optional): Expected date in the input files. If not set,
            all files will be used. This might cause unexpected behavior if
            there are files from several days. If date is set as 'YYYY-MM-DD',
            only files that match the date will be used.

    Returns:
        tuple: 2-element tuple containing

        - str: UUID of the generated file.
        - list: Files used in the processing.

    Raises:
        RuntimeError: Failed to read the binary data.

    Examples:
        >>> from cloudnetpy.instruments import hatpro2nc
        >>> site_meta = {'name': 'Hyytiala', 'altitude': 174}
        >>> hatpro2nc('/path/to/files/', 'hatpro.nc', site_meta)

    """
    filenames = utils.get_sorted_filenames(path_to_lwp_files, '.LWP')
    one_day_of_data, valid_files = _create_one_day_data_record(filenames, date)
    if not valid_files:
        return '', []
    hatpro = rpg.Rpg(one_day_of_data, site_meta, 'RPG-HATPRO')
    attributes = output.add_time_attribute(ATTRIBUTES, hatpro.date)
    output.update_attributes(hatpro.data, attributes)
    return _save_hatpro(hatpro, output_file, valid_files, keep_uuid, uuid)


def _create_one_day_data_record(l1_files: list, date: Union[str, None]) -> Tuple[dict, list]:
    """Concatenates all HATPRO data from one day."""
    hatpro_objects, valid_files = _get_hatpro_objects(l1_files, date)
    data, header = rpg.stack_rpg_data(hatpro_objects)
    if len(valid_files) > 1:
        try:
            header = rpg.reduce_header(header)
        except AssertionError as error:
            raise RuntimeError(error)
    data = rpg.mask_invalid_data(data)
    return {**header, **data}, valid_files


def _get_hatpro_objects(files: list, expected_date: Union[str, None]) -> Tuple[list, list]:
    """Creates a list of HATPRO objects from the file names."""
    objects = []
    valid_files = []
    for file in files:
        try:
            obj = HatproBin(file)
            if expected_date is not None:
                obj = _validate_date(obj, expected_date)
        except (TypeError, ValueError) as err:
            print(err)
            continue
        objects.append(obj)
        valid_files.append(file)
    return objects, valid_files


def _validate_date(obj, expected_date: str):
    if obj.header['_lwp_time_ref'] == 0:
        raise ValueError('Can not validate non-UTC dates.')
    inds = []
    for ind, timestamp in enumerate(obj.data['time'][:]):
        date = '-'.join(utils.seconds2date(timestamp)[:3])
        if date == expected_date:
            inds.append(ind)
    if not inds:
        raise ValueError('No profiles matching the expected date.')
    for key in obj.data.keys():
        obj.data[key] = obj.data[key][inds]
    return obj


class HatproBin:
    """HATPRO binary file reader."""
    def __init__(self, filename):
        self.filename = filename
        self._file_position = 0
        self.header = self.read_header()
        self.data = self.read_data()

    def read_header(self) -> dict:
        """Reads the header."""

        def append(names: tuple, dtype: type = np.int32, n_values: int = 1) -> None:
            """Updates header dictionary."""
            for name in names:
                header[name] = np.fromfile(file, dtype, int(n_values))

        header = {}
        file = open(self.filename, 'rb')
        append(('file_code',
                '_n_samples'), np.int32)
        append(('_lwp_min',
                '_lwp_max',), np.float32)
        append(('_lwp_time_ref',
                'lwp_retrieval'), np.int32)
        self._file_position = file.tell()
        file.close()
        return header

    def read_data(self) -> dict:
        """Reads the data."""
        file = open(self.filename, 'rb')
        file.seek(self._file_position)

        data = {
            'time': np.zeros(self.header['_n_samples'], dtype=np.int32),
            'rain_flag': np.zeros(self.header['_n_samples'], dtype=np.int32),
            'lwp': np.zeros(self.header['_n_samples']),
            'lwp_angle': np.zeros(self.header['_n_samples'])
        }

        for sample in range(self.header['_n_samples'][0]):
            data['time'][sample] = np.fromfile(file, np.int32, 1)
            data['rain_flag'][sample] = np.fromfile(file, np.int8, 1)
            data['lwp'][sample] = np.fromfile(file, np.float32, 1)
            data['lwp_angle'][sample] = np.fromfile(file, np.float32, 1)

        file.close()
        return data


def _save_hatpro(hatpro: rpg.Rpg,
                 output_file: str,
                 valid_files: list,
                 keep_uuid: bool,
                 uuid: Union[str, None] = None) -> Tuple[str, list]:
    """Saves the HATPRO file."""

    dims = {'time': len(hatpro.data['time'][:])}

    file_type = 'mwr'
    rootgrp = output.init_file(output_file, dims, hatpro.data, keep_uuid, uuid)
    file_uuid = rootgrp.file_uuid
    output.add_file_type(rootgrp, file_type)
    rootgrp.title = f"{file_type.capitalize()} file from {hatpro.location}"
    rootgrp.year, rootgrp.month, rootgrp.day = hatpro.date
    rootgrp.location = hatpro.location
    rootgrp.history = f"{utils.get_time()} - {file_type} file created"
    rootgrp.source = hatpro.source
    output.add_references(rootgrp)
    rootgrp.close()
    return file_uuid, valid_files


DEFINITIONS = {
    'lwp_retrieval':
        ('\n'
         'Value 0: Linear Regression\n'
         'Value 1: Quadratic Regression\n'
         'Value 2: Neural Network'),
    'rain_flag':
        ('\n'
         'Bit 0: Rain information (0=no rain, 1=raining)\n'
         'Bit 1/2: Quality level (0=Not evaluated, 1=high, 2=medium, 3=low)\n'
         'Bit 3/4: Reason for reduced quality'),
 }

ATTRIBUTES = {
    'file_code': MetaData(
        long_name='File code',
        comment='RPG HATPRO software version.',
    ),
    'program_number': MetaData(
        long_name='Program number',
    ),
    'lwp_retrieval': MetaData(
        long_name='Retrieval method',
        definition=DEFINITIONS['lwp_retrieval']
    ),
    'lwp': MetaData(
        long_name='Liquid water path',
        units='g m-2'
    ),
    'rain_flag': MetaData(
        long_name='Rain flag.',
        definition=DEFINITIONS['rain_flag'],
        comment='Rain information as 8 bit array. See RPG HATPRO manual for more information.'
    )

}
