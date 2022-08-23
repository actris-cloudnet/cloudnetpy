"""This module contains RPG Cloud Radar related functions."""
import logging
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Union

from cloudnetpy import output, utils
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import general, rpg
from cloudnetpy.instruments.rpg_reader import (
    HatproBin,
    HatproBinCombined,
    HatproBinIwv,
    HatproBinLwp,
)


def hatpro2nc(
    path_to_files: str,
    output_file: str,
    site_meta: dict,
    uuid: Optional[str] = None,
    date: Optional[str] = None,
) -> Tuple[str, list]:
    """Converts RPG HATPRO microwave radiometer data into Cloudnet Level 1b
    netCDF file.

    This function reads one day of RPG HATPRO .LWP and .IWV binary files,
    concatenates the data and writes it into netCDF file.

    Args:
        path_to_files: Folder containing one day of RPG HATPRO files.
        output_file: Output file name.
        site_meta: Dictionary containing information about the site with keys:

            - `name`: Name of the site (required)
            - `altitude`: Site altitude in [m] (optional).
            - `latitude` (optional).
            - `longitude` (optional).

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
        ValidTimeStampError: No valid timestamps found.

    Examples:
        >>> from cloudnetpy.instruments import hatpro2nc
        >>> site_meta = {'name': 'Hyytiala', 'altitude': 174}
        >>> hatpro2nc('/path/to/files/', 'hatpro.nc', site_meta)

    """
    hatpro_objects, valid_files = _get_hatpro_objects(Path(path_to_files), date)
    if not valid_files:
        raise ValidTimeStampError
    one_day_of_data = rpg.create_one_day_data_record(hatpro_objects)
    hatpro = rpg.Hatpro(one_day_of_data, site_meta)
    hatpro.sort_timestamps()
    hatpro.convert_time_to_fraction_hour("float64")
    general.add_site_geolocation(hatpro)
    hatpro.remove_duplicate_timestamps()
    attributes = output.add_time_attribute({}, hatpro.date)
    output.update_attributes(hatpro.data, attributes)
    uuid = output.save_level1b(hatpro, output_file, uuid)
    return uuid, valid_files


def _get_hatpro_objects(
    directory: Path, expected_date: Union[str, None]
) -> Tuple[List[HatproBinCombined], List[str]]:
    objects = defaultdict(list)
    for filename in directory.iterdir():
        try:
            obj: HatproBin
            extension = filename.suffix.upper()
            if extension == ".LWP":
                obj = HatproBinLwp(filename)
            elif extension == ".IWV":
                obj = HatproBinIwv(filename)
            else:
                continue
            obj.screen_bad_profiles()
            if expected_date is not None:
                obj = _validate_date(obj, expected_date)
            objects[filename.stem].append(obj)
        except (TypeError, ValueError) as err:
            logging.warning(f"Ignoring file '{filename}': {err}")
            continue

    valid_files: List[str] = []
    combined_objs = []
    for _stem, objs in sorted(objects.items()):
        try:
            combined_objs.append(HatproBinCombined(objs))
            valid_files.extend(str(obj.filename) for obj in objs)
        except (TypeError, ValueError) as err:
            files = "'" + "', '".join(str(obj.filename) for obj in objs) + "'"
            logging.warning(f"Ignoring files {files}: {err}")
            continue

    return combined_objs, valid_files


def _validate_date(obj: HatproBin, expected_date: str):
    if obj.header["_time_reference"] != 1:
        raise ValueError("Can not validate non-UTC dates")
    inds = []
    for ind, timestamp in enumerate(obj.data["time"][:]):
        date = "-".join(utils.seconds2date(timestamp)[:3])
        if date == expected_date:
            inds.append(ind)
    if not inds:
        raise ValueError("Timestamps not what expected")
    obj.data = obj.data[:][inds]
    return obj
