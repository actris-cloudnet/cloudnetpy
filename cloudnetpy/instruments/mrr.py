import datetime
import logging
import re
from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from uuid import UUID

import netCDF4

from cloudnetpy import concat_lib, output, utils
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.nc_radar import NcRadar
from cloudnetpy.metadata import MetaData


def mrr2nc(
    input_file: PathLike | str | Iterable[PathLike | str],
    output_file: PathLike | str,
    site_meta: dict,
    uuid: UUID | str | None = None,
    date: datetime.date | str | None = None,
) -> str:
    """Converts METEK MRR-PRO data into Cloudnet Level 1b netCDF file.

    This function converts raw MRR file(s) into a much smaller file that
    contains only the relevant data.

    Args:
        input_file: Filename of a daily MMR-PRO .nc file, path to directory
            containing several non-concatenated .nc files from one day, or list
            of filenames.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            value pairs are `name`, `latitude`, `longitude` and `altitude`.
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    Examples:
          >>> from cloudnetpy.instruments import mira2nc
          >>> site_meta = {'name': 'LIM', 'latitude': 51.333, 'longitude': 12.389}
          >>> mrr2nc('input.nc', 'output.nc', site_meta)
    """
    if isinstance(uuid, str):
        uuid = UUID(uuid)
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)

    keymap = {
        "RR": "rainfall_rate",
        "WIDTH": "width",
        "VEL": "v",
        "LWC": "lwc",
        "Ze": "Zh",
        "PIA": "pia",
    }

    def valid_nc_files(files: Iterable[PathLike | str]) -> Iterable[PathLike | str]:
        for file in files:
            try:
                with netCDF4.Dataset(file):
                    yield file
            except OSError:
                logging.warning("Skipping invalid file: %s", file)

    def concat_files(dir_name: str, files: Iterable[PathLike | str]) -> str:
        with NamedTemporaryFile(
            dir=dir_name,
            suffix=".nc",
            delete=False,
        ) as temp_file:
            tmp_filename = temp_file.name
            variables = [*keymap.keys(), "elevation"]
            valid_files = list(valid_nc_files(files))
            concat_lib.concatenate_files(
                valid_files,
                tmp_filename,
                variables=variables,
                ignore=["time_coverage_start", "time_coverage_end"],
            )
            return tmp_filename

    with TemporaryDirectory() as temp_dir:
        if isinstance(input_file, PathLike | str):
            path = Path(input_file)
            if path.is_dir():
                input_file = concat_files(
                    temp_dir,
                    (p for p in path.iterdir() if p.suffix.lower() == ".nc"),
                )
        else:
            input_file = concat_files(temp_dir, input_file)

        with MrrPro(input_file, site_meta) as mrr:
            mrr.init_data(keymap)
            mrr.fix_units()
            mrr.date = mrr.init_date()
            if date:
                mrr.screen_by_date(date)
            mrr.add_time_and_range()
            mrr.add_site_geolocation()
            mrr.add_zenith_angle()
            mrr.add_radar_specific_variables()
            mrr.add_height()
            mrr.sort_timestamps()
        attributes = output.add_time_attribute(ATTRIBUTES, mrr.date)
        output.update_attributes(mrr.data, attributes)
        return output.save_level1b(mrr, output_file, uuid)


class MrrPro(NcRadar):
    """Class for MRR-PRO raw data. Child of NcRadar().

    Args:
        full_path: MRR-PRO netCDF filename.
        site_meta: Site properties in a dictionary. Required keys are `name`,
            `latitude`, `longitude` and `altitude`.

    """

    epoch = (1970, 1, 1)

    def __init__(self, full_path: PathLike | str, site_meta: dict):
        super().__init__(full_path, site_meta)
        self.instrument = instruments.MRR_PRO
        if m := re.search(
            r"serial number:\s*(\w+)",
            self.dataset.instrument_name,
            re.IGNORECASE,
        ):
            self.serial_number = m[1]

    def init_date(self) -> list[str]:
        time_stamps = self.getvar("time")
        return utils.seconds2date(time_stamps[0], (1970, 1, 1))[:3]

    def fix_units(self) -> None:
        self.data["v"].data *= -1  # towards -> away from instrument
        self.data["rainfall_rate"].data /= 3600000  # mm h-1 -> m s-1
        self.data["lwc"].data *= 0.001  # g m-3 -> kg m-3

    def add_zenith_angle(self) -> None:
        elevation = self.getvar("elevation")
        zenith = 90 - elevation
        self.append_data(zenith, "zenith_angle")

    def screen_by_date(self, expected_date: datetime.date) -> None:
        """Screens incorrect time stamps."""
        time_stamps = self.getvar("time")
        valid_indices = []
        for ind, timestamp in enumerate(time_stamps):
            date = "-".join(utils.seconds2date(timestamp, self.epoch)[:3])
            if date == expected_date.isoformat():
                valid_indices.append(ind)
        self.screen_time_indices(valid_indices)


ATTRIBUTES = {
    "lwc": MetaData(long_name="Liquid water content", units="kg m-3"),
    "pia": MetaData(long_name="Path integrated rain attenuation", units="dB"),
}
