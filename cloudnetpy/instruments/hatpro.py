"""This module contains RPG Cloud Radar related functions."""
from typing import Union, Tuple, Optional
import numpy as np
from cloudnetpy import utils, output
from cloudnetpy.metadata import MetaData
from cloudnetpy.instruments import rpg


def hatpro2nc(path_to_lwp_files: str,
              output_file: str,
              site_meta: dict,
              keep_uuid: Optional[bool] = False,
              uuid: Optional[str] = None,
              date: Optional[str] = None) -> Tuple[str, list]:
    """Converts RPG HATPRO microwave radiometer data (LWP) into Cloudnet Level 1b netCDF file.

    This function reads one day of RPG HATPRO .LWP binary files,
    concatenates the data and writes it into netCDF file.

    Args:
        path_to_lwp_files: Folder containing one day of RPG HATPRO files.
        output_file: Output file name.
        site_meta: Dictionary containing information about the
            site. Required key value pairs are `altitude` (metres above mean
            sea level) and `name`.
        keep_uuid: If True, keeps the UUID of the old file,
            if that exists. Default is False when new UUID is generated.
        uuid: Set specific UUID for the file.
        date: Expected date in the input files. If not set,
            all files will be used. This might cause unexpected behavior if
            there are files from several days. If date is set as 'YYYY-MM-DD',
            only files that match the date will be used.

    Returns:
        2-element tuple containing

        - UUID of the generated file.
        - Files used in the processing.

    Raises:
        RuntimeError: Failed to read the binary data.

    Examples:
        >>> from cloudnetpy.instruments import hatpro2nc
        >>> site_meta = {'name': 'Hyytiala', 'altitude': 174}
        >>> hatpro2nc('/path/to/files/', 'hatpro.nc', site_meta)

    """
    all_files = utils.get_sorted_filenames(path_to_lwp_files, '.LWP')
    rpg_objects, valid_files = _get_rpg_objects(all_files, date)
    one_day_of_data = rpg.create_one_day_data_record(rpg_objects)
    if not valid_files:
        return '', []
    hatpro = rpg.Rpg(one_day_of_data, site_meta, 'RPG-HATPRO')
    attributes = output.add_time_attribute(ATTRIBUTES, hatpro.date)
    output.update_attributes(hatpro.data, attributes)
    return rpg.save_rpg(hatpro, output_file, valid_files, keep_uuid, uuid)


def _get_rpg_objects(files: list, expected_date: Union[str, None]) -> Tuple[list, list]:
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
    if obj.header['_time_reference'] == 0:
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
        file = open(self.filename, 'rb')
        header = {
            'file_code': np.fromfile(file, np.int32, 1),
            '_n_samples': np.fromfile(file, np.int32, 1),
            '_lwp_min_max': np.fromfile(file, np.float32, 2),
            '_time_reference': np.fromfile(file, np.int32, 1),
            'retrieval_method': np.fromfile(file, np.int32, 1)
        }
        self._file_position = file.tell()
        file.close()
        return header

    def read_data(self) -> dict:
        """Reads the data."""
        file = open(self.filename, 'rb')
        file.seek(self._file_position)

        data = {
            'time': np.zeros(self.header['_n_samples'], dtype=np.int32),
            'quality_flag': np.zeros(self.header['_n_samples'], dtype=np.int32),
            'lwp': np.zeros(self.header['_n_samples']),
            'zenith': np.zeros(self.header['_n_samples'], dtype=np.float32)
        }

        version = self._get_hatpro_version()
        angle_dtype = np.float32 if version == 1 else np.int32
        data['_instrument_angles'] = np.zeros(self.header['_n_samples'], dtype=angle_dtype)

        for sample in range(self.header['_n_samples'][0]):
            data['time'][sample] = np.fromfile(file, np.int32, 1)
            data['quality_flag'][sample] = np.fromfile(file, np.int8, 1)
            data['lwp'][sample] = np.fromfile(file, np.int32, 1)
            data['_instrument_angles'][sample] = np.fromfile(file, angle_dtype, 1)

        data = _add_zenith(version, data)

        file.close()
        return data

    def _get_hatpro_version(self) -> int:
        if self.header['file_code'][0] == 934501978:
            return 1
        if self.header['file_code'][0] == 934501000:
            return 2
        raise ValueError(f'Unknown HATPRO version. {self.header["file_code"][0]}')


def _add_zenith(version: int, data: dict) -> dict:
    if version == 1:
        del data['zenith']  # Impossible to understand how zenith is decoded in the values
    else:
        data['zenith'] = np.array([int(str(x)[:5])/1000 for x in data['_instrument_angles']])
    return data


DEFINITIONS = {
    'retrieval_method':
        ('\n'
         'Value 0: Linear Regression\n'
         'Value 1: Quadratic Regression\n'
         'Value 2: Neural Network'),
    'quality_flag':
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
    'retrieval_method': MetaData(
        long_name='Retrieval method',
        definition=DEFINITIONS['retrieval_method']
    ),
    'lwp': MetaData(
        long_name='Liquid water path',
        units='g m-2'
    ),
    'zenith': MetaData(
        long_name='Zenith angle',
        units='degrees',
    ),
    'quality_flag': MetaData(
        long_name='Quality flag.',
        definition=DEFINITIONS['quality_flag'],
        comment='Quality information as an 8 bit array. See RPG HATPRO manual for more information.'
    )

}
