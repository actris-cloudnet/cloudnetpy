"""Module for reading / converting BASTA radar data."""

import numpy as np

from cloudnetpy import output
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.nc_radar import NcRadar
from cloudnetpy.metadata import MetaData


def basta2nc(
    basta_file: str,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | None = None,
) -> str:
    """Converts BASTA cloud radar data into Cloudnet Level 1b netCDF file.

    This function converts daily BASTA file into a much smaller file that
    contains only the relevant data and can be used in further processing
    steps.

    Args:
        basta_file: Filename of a daily BASTA .nc file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
        UUID of the generated file.

    Raises:
        ValueError: Timestamps do not match the expected date.

    Examples:
          >>> from cloudnetpy.instruments import basta2nc
          >>> site_meta = {'name': 'Palaiseau', 'latitude': 48.718, 'longitude': 2.207}
          >>> basta2nc('basta_file.nc', 'radar.nc', site_meta)

    """
    keymap = {
        "reflectivity": "Zh",
        "velocity": "v",
        "radar_pitch": "radar_pitch",
        "radar_yaw": "radar_yaw",
        "radar_roll": "radar_roll",
    }

    with Basta(basta_file, site_meta) as basta:
        basta.init_data(keymap)
        if date is not None:
            basta.validate_date(date)
        basta.screen_data(keymap)
        basta.add_time_and_range(time_dtype="f8")
        basta.add_site_geolocation()
        basta.add_zenith_angle()
        basta.add_radar_specific_variables()
        basta.add_height()
        basta.sort_timestamps()
        basta.remove_duplicate_timestamps()
        basta.test_if_all_masked()
    attributes = output.add_time_attribute(ATTRIBUTES, basta.date)
    output.update_attributes(basta.data, attributes)
    return output.save_level1b(basta, output_file, uuid)


class Basta(NcRadar):
    """Class for BASTA raw radar data. Child of NcRadar().

    Args:
        full_path: BASTA netCDF filename.
        site_meta: Site properties in a dictionary. Required key is `name`.

    """

    def __init__(self, full_path: str, site_meta: dict):
        super().__init__(full_path, site_meta)
        self.date: list[str] = self.get_date()
        self.instrument = instruments.BASTA

    def screen_data(self, keymap: dict) -> None:
        """Saves only valid pixels."""
        mask = self.getvar("background_mask")
        for key in keymap.values():
            if key in self.data and self.data[key].data.ndim == mask.ndim:
                self.data[key].mask_indices(np.where(mask != 1))

    def validate_date(self, expected_date: str) -> None:
        """Validates expected data."""
        date_units = self.dataset.variables["time"].units
        date = date_units.split()[2]
        if expected_date != date:
            raise ValidTimeStampError

    def add_zenith_angle(self) -> None:
        elevation = self.getvar("elevation")
        zenith = 90 - elevation
        self.append_data(zenith, "zenith_angle")


ATTRIBUTES = {
    "radar_pitch": MetaData(
        long_name="Radar pitch angle",
        units="degree",
        standard_name="platform_roll",
    ),
    "radar_yaw": MetaData(
        long_name="Radar yaw angle",
        units="degree",
        standard_name="platform_yaw",
    ),
    "radar_roll": MetaData(
        long_name="Radar roll angle",
        units="degree",
        standard_name="platform_roll",
    ),
}
