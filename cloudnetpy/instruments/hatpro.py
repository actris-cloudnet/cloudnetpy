"""This module contains RPG Cloud Radar related functions."""
from typing import Union, Tuple, Optional
import logging
from cloudnetpy import utils, output
from cloudnetpy.metadata import MetaData
from cloudnetpy.instruments import rpg
from cloudnetpy.instruments.rpg_reader import HatproBin


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
    hatpro_objects, valid_files = _get_hatpro_objects(all_files, date)
    one_day_of_data = rpg.create_one_day_data_record(hatpro_objects)
    if not valid_files:
        return '', []
    hatpro = rpg.Rpg(one_day_of_data, site_meta, 'RPG-HATPRO')
    output.update_attributes(hatpro.data, ATTRIBUTES)
    return rpg.save_rpg(hatpro, output_file, valid_files, keep_uuid, uuid)


def _get_hatpro_objects(files: list, expected_date: Union[str, None]) -> Tuple[list, list]:
    """Creates a list of HATPRO objects from the file names."""
    objects = []
    valid_files = []
    for file in files:
        try:
            obj = HatproBin(file)
            obj.screen_bad_profiles()
            if expected_date is not None:
                obj = _validate_date(obj, expected_date)
        except (TypeError, ValueError) as err:
            logging.warning(err)
            continue
        objects.append(obj)
        valid_files.append(file)
    return objects, valid_files


def _validate_date(obj, expected_date: str):
    if obj.header['_time_reference'] == 0:
        raise ValueError('Ignoring a file (can not validate non-UTC dates)')
    inds = []
    for ind, timestamp in enumerate(obj.data['time'][:]):
        date = '-'.join(utils.seconds2date(timestamp)[:3])
        if date == expected_date:
            inds.append(ind)
    if not inds:
        raise ValueError('Ignoring a file (time stamps not what expected)')
    for key in obj.data.keys():
        obj.data[key] = obj.data[key][inds]
    return obj


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
    'time': MetaData(
        units='seconds since 2001-01-01 00:00:00',
        long_name='sample time',
    ),
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
    'LWP': MetaData(
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
