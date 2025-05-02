"""This module contains RPG Cloud Radar related functions."""

import datetime
import logging
from collections import defaultdict
from pathlib import Path
from typing import Literal

import netCDF4
import numpy as np
from mwrpy.exceptions import MissingInputData
from mwrpy.level1.lev1_meta_nc import ATTRIBUTES_1B01
from mwrpy.level1.write_lev1_nc import lev1_to_nc
from mwrpy.version import __version__ as mwrpy_version
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.exceptions import HatproDataError, ValidTimeStampError
from cloudnetpy.instruments import rpg
from cloudnetpy.instruments.instruments import HATPRO, LHATPRO, LHUMPRO_U90, Instrument
from cloudnetpy.instruments.rpg_reader import (
    HatproBin,
    HatproBinCombined,
    HatproBinIwv,
    HatproBinLwp,
)

IType = Literal["hatpro", "lhatpro", "lhumpro_u90"]
ITYPE_MAP: dict[IType, Instrument] = {
    "hatpro": HATPRO,
    "lhatpro": LHATPRO,
    "lhumpro_u90": LHUMPRO_U90,
}


def hatpro2l1c(
    mwr_dir: str,
    output_file: str,
    site_meta: dict,
    instrument_type: IType = "hatpro",
    uuid: str | None = None,
    date: datetime.date | str | None = None,
) -> str:
    """Converts RPG HATPRO microwave radiometer data into Cloudnet Level 1c netCDF file.

    Args:
        mwr_dir: Folder containing one day of HATPRO files.
        output_file: Output file name.
        site_meta: Dictionary containing information about the site and instrument
        instrument_type: Specific type of the RPG microwave radiometer.
        uuid: Set specific UUID for the file.
        date: Expected date in the input files.

    Returns:
        UUID of the generated file.
    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)

    coeff_files = site_meta.get("coefficientFiles")
    time_offset = site_meta.get("time_offset")

    try:
        hatpro_raw = lev1_to_nc(
            "1C01",
            mwr_dir,
            instrument_type=instrument_type,
            output_file=output_file,
            coeff_files=coeff_files,
            instrument_config=site_meta,
            date=date,
            time_offset=time_offset,
        )
    except MissingInputData as err:
        raise HatproDataError(str(err)) from err

    hatpro = HatproL1c(hatpro_raw, site_meta, ITYPE_MAP[instrument_type])

    flags = hatpro.data["quality_flag"][:]
    bad_percentage = ma.sum(flags != 0) / flags.size * 100
    if bad_percentage > 90:
        msg = "More than 90% of brightness temperatures are flagged"
        raise HatproDataError(msg)

    timestamps = hatpro.data["time"][:]
    if date is not None:
        # Screen timestamps if these assertions start to fail
        if not np.all(np.diff(timestamps) > 0):
            msg = "Timestamps are not increasing"
            raise RuntimeError(msg)
        dates = [
            datetime.datetime.fromtimestamp(t, tz=datetime.timezone.utc).date()
            for t in timestamps
        ]
        if len(set(dates)) != 1:
            msg = f"Several dates, something is wrong: {set(dates)}"
            raise RuntimeError(msg)
        if date != dates[0]:
            msg = f"Expected date {date}, got {dates[0]}"
            raise RuntimeError(msg)

    decimal_hours = utils.seconds2hours(timestamps)
    hatpro.data["time"] = CloudnetArray(decimal_hours, "time", data_type="f8")
    hatpro.data.pop("time_bnds")
    hatpro.data["t_amb"].dimensions = ("time", "t_amb_nb")

    for key in ("elevation_angle", "ir_elevation_angle"):
        if key not in hatpro.data:
            continue
        zenith_angle = 90 - hatpro.data[key][:]
        new_key = key.replace("elevation", "zenith")
        hatpro.data[new_key] = CloudnetArray(zenith_angle, new_key)

    if "ir_wavelength" in hatpro.data:
        hatpro.data["ir_wavelength"].dimensions = ("ir_channel",)
    if "irt" in hatpro.data:
        hatpro.data["irt"].dimensions = ("time", "ir_channel")

    utils.add_site_geolocation(hatpro.data, gps=True, site_meta=site_meta)

    attrs_copy = ATTRIBUTES_1B01.copy()
    attributes = output.add_time_attribute(attrs_copy, hatpro.date)
    output.update_attributes(hatpro.data, attributes)
    uuid = output.save_level1b(hatpro, output_file, uuid)
    with netCDF4.Dataset(output_file, "a") as nc:
        nc.cloudnet_file_type = "mwr-l1c"
        nc.title = nc.title.replace("radiometer", "radiometer Level 1c")
        nc.mwrpy_version = mwrpy_version
        nc.mwrpy_coefficients = ", ".join(site_meta["coefficientLinks"])
        nc.history = nc.history.replace("mwr", "mwr-l1c")

    return uuid


class HatproL1c:
    def __init__(self, hatpro, site_meta: dict, instrument: Instrument):
        self.raw_data = hatpro.raw_data
        self.data = hatpro.data
        self.date = hatpro.date.isoformat().split("-")
        self.site_meta = site_meta
        self.instrument = instrument


def hatpro2nc(
    path_to_files: str,
    output_file: str,
    site_meta: dict,
    instrument_type: IType = "hatpro",
    uuid: str | None = None,
    date: str | None = None,
) -> tuple[str, list]:
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

        instrument_type: Specific type of the RPG microwave radiometer.
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
    is_lwp_files = any(f.endswith(".LWP") for f in valid_files)
    is_iwv_files = any(f.endswith(".IWV") for f in valid_files)
    if not is_lwp_files:
        raise ValidTimeStampError
    if is_iwv_files:
        _add_missing_variables(hatpro_objects, ("lwp", "iwv"))
    one_day_of_data = rpg.create_one_day_data_record(hatpro_objects)
    hatpro = rpg.Hatpro(one_day_of_data, site_meta, ITYPE_MAP[instrument_type])
    hatpro.add_site_geolocation()
    hatpro.convert_time_to_fraction_hour("float64")
    hatpro.sort_timestamps()
    hatpro.remove_duplicate_timestamps()
    attributes = output.add_time_attribute({}, hatpro.date)
    output.update_attributes(hatpro.data, attributes)
    uuid = output.save_level1b(hatpro, output_file, uuid)
    return uuid, valid_files


def _get_hatpro_objects(
    directory: Path,
    expected_date: str | None,
) -> tuple[list[HatproBinCombined], list[str]]:
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
        except (TypeError, ValueError, ValidTimeStampError) as err:
            logging.warning("Ignoring file '%s': %s", filename, err)
            continue

    valid_files: list[str] = []
    combined_objs = []
    for _stem, objs in sorted(objects.items()):
        try:
            combined_objs.append(HatproBinCombined(objs))
            valid_files.extend(str(obj.filename) for obj in objs)
        except (TypeError, ValueError) as err:
            files = "'" + "', '".join(str(obj.filename) for obj in objs) + "'"
            logging.warning("Ignoring files %s: %s", files, err)
            continue

    return combined_objs, valid_files


def _validate_date(obj: HatproBin, expected_date: str) -> HatproBin:
    if obj.header["_time_reference"] != 1:
        msg = "Can not validate non-UTC dates"
        raise ValueError(msg)
    inds = []
    for ind, timestamp in enumerate(obj.data["time"][:]):
        date = "-".join(utils.seconds2date(timestamp)[:3])
        if date == expected_date:
            inds.append(ind)
    if not inds:
        msg = f"No valid timestamps found for date {expected_date}"
        raise ValueError(msg)
    obj.data = obj.data[:][inds]
    return obj


def _add_missing_variables(
    hatpro_objects: list[HatproBinCombined],
    keys: tuple,
) -> list[HatproBinCombined]:
    for obj in hatpro_objects:
        for key in keys:
            if key not in obj.data:
                obj.data[key] = ma.masked_all((len(obj.data["time"]),))
    return hatpro_objects
