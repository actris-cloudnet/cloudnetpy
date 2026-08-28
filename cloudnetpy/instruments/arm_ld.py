"""Module for reading ARM laser disdrometer (OTT Parsivel2) data."""

import datetime
from os import PathLike
from uuid import UUID

import netCDF4
import numpy as np
import numpy.typing as npt

from cloudnetpy.disdronator.parsivel import read_parsivel_l1
from cloudnetpy.instruments.arm_utils import read_geolocation
from cloudnetpy.instruments.disdrometer import (
    ATTRIBUTES,
    Parsivel,
    _process_disdrometer,
)
from cloudnetpy.utils import get_epoch


def armld2nc(
    raw_file: str | PathLike,
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts ARM laser disdrometer (ld) data into Cloudnet Level 1b netCDF
    file.

    The ARM `ld.b1` datastream contains OTT Parsivel2 measurements ingested
    into netCDF. The raw particle size / velocity spectrum is processed like
    the raw Parsivel telegrams.

    Args:
        raw_file: Daily ARM `ld` netCDF file, e.g. `sgpldC1.b1.20220601.000000.cdf`.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            value pair is `name`. Optional are `latitude`, `longitude` and
            `altitude` (taken from the raw file if missing).
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Examples:
          >>> from cloudnetpy.instruments import armld2nc
          >>> site_meta = {'name': 'Southern Great Plains'}
          >>> armld2nc('sgpldC1.b1.20220601.000000.cdf', 'disdrometer.nc', site_meta)

    """
    site_meta = _add_geolocation_from_file(raw_file, site_meta)
    return _process_disdrometer(
        Parsivel,
        _read_arm_ld,
        read_parsivel_l1,
        ATTRIBUTES,
        raw_file,
        output_file,
        site_meta,
        uuid,
        date,
    )


def _read_arm_ld(
    filename: str | PathLike,
) -> tuple[npt.NDArray, dict[int, npt.NDArray]]:
    """Reads ARM ld file into Parsivel telegram fields."""
    with netCDF4.Dataset(filename) as nc:
        epoch = np.datetime64(get_epoch(nc["time"].units).replace(tzinfo=None), "s")
        seconds = np.round(np.array(nc["time"][:], dtype=float)).astype(int)
        time = (epoch + seconds.astype("timedelta64[s]")).astype(datetime.datetime)
        data: dict[int, npt.NDArray] = {
            93: np.ma.filled(nc["raw_spectrum"][:], 0).astype(np.int32),
        }
        if "weather_code" in nc.variables:
            data[3] = np.ma.filled(nc["weather_code"][:], -9999).astype(np.int32)
        serial = getattr(nc, "serial_number", None)
        if serial:
            data[13] = np.full(len(time), str(serial))
    return np.array(time), data


def _add_geolocation_from_file(raw_file: str | PathLike, site_meta: dict) -> dict:
    with netCDF4.Dataset(raw_file) as nc:
        return read_geolocation(nc, site_meta)
