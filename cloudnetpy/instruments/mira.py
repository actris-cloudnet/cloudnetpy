"""Module for reading raw cloud radar data."""
import os
from tempfile import TemporaryDirectory
from typing import List, Optional

from cloudnetpy import concat_lib, output, utils
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import general
from cloudnetpy.instruments.instruments import MIRA35
from cloudnetpy.instruments.nc_radar import NcRadar
from cloudnetpy.metadata import MetaData


def mira2nc(
    raw_mira: str,
    output_file: str,
    site_meta: dict,
    uuid: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """Converts METEK MIRA-35 cloud radar data into Cloudnet Level 1b netCDF file.

    This function converts raw MIRA file(s) into a much smaller file that
    contains only the relevant data and can be used in further processing
    steps.

    Args:
        raw_mira: Filename of a daily MIRA .mmclx file. Can be also a folder containing several
            non-concatenated .mmclx files from one day.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key value pair
            is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    Examples:
          >>> from cloudnetpy.instruments import mira2nc
          >>> site_meta = {'name': 'Vehmasmaki'}
          >>> mira2nc('raw_radar.mmclx', 'radar.nc', site_meta)
          >>> mira2nc('/one/day/of/mira/mmclx/files/', 'radar.nc', site_meta)

    """
    keymap = {
        "Zg": "Zh",
        "VELg": "v",
        "RMSg": "width",
        "LDRg": "ldr",
        "SNRg": "SNR",
        "elv": "elevation",
        "azi": "azimuth_angle",
        "aziv": "azimuth_velocity",
        "nfft": "nfft",
        "nave": "nave",
        "prf": "prf",
        "rg0": "rg0",
    }

    with TemporaryDirectory() as temp_dir:
        if os.path.isdir(raw_mira):
            mmclx_filename = f"{temp_dir}/tmp.mmclx"
            valid_filenames = utils.get_sorted_filenames(raw_mira, ".mmclx")
            valid_filenames = general.get_files_with_common_range(valid_filenames)
            variables = list(keymap.keys())
            concat_lib.concatenate_files(valid_filenames, mmclx_filename, variables=variables)
        else:
            mmclx_filename = raw_mira

        with Mira(mmclx_filename, site_meta) as mira:
            mira.init_data(keymap)
            if date is not None:
                mira.screen_by_date(date)
                mira.date = date.split("-")
            mira.sort_timestamps()
            mira.remove_duplicate_timestamps()
            general.linear_to_db(mira, ("Zh", "ldr", "SNR"))
            mira.screen_by_snr()
            mira.mask_invalid_data()
            mira.add_time_and_range()
            general.add_site_geolocation(mira)
            general.add_radar_specific_variables(mira)
            valid_indices = mira.add_zenith_and_azimuth_angles()
            general.screen_time_indices(mira, valid_indices)
            general.add_height(mira)
        attributes = output.add_time_attribute(ATTRIBUTES, mira.date)
        output.update_attributes(mira.data, attributes)
        uuid = output.save_level1b(mira, output_file, uuid)
        return uuid


class Mira(NcRadar):
    """Class for MIRA-35 raw radar data. Child of NcRadar().

    Args:
        full_path: Filename of a daily MIRA .mmclx NetCDF file.
        site_meta: Site properties in a dictionary. Required keys are: `name`.

    """

    epoch = (1970, 1, 1)

    def __init__(self, full_path: str, site_meta: dict):
        super().__init__(full_path, site_meta)
        self.date = self._init_mira_date()
        self.instrument = MIRA35

    def screen_by_date(self, expected_date: str) -> None:
        """Screens incorrect time stamps."""
        time_stamps = self.getvar("time")
        valid_indices = []
        for ind, timestamp in enumerate(time_stamps):
            date = "-".join(utils.seconds2date(timestamp, self.epoch)[:3])
            if date == expected_date:
                valid_indices.append(ind)
        if not valid_indices:
            raise ValidTimeStampError
        general.screen_time_indices(self, valid_indices)

    def _init_mira_date(self) -> List[str]:
        time_stamps = self.getvar("time")
        return utils.seconds2date(time_stamps[0], self.epoch)[:3]


ATTRIBUTES = {
    "nfft": MetaData(
        long_name="Number of FFT points",
        units="1",
    ),
    "nave": MetaData(
        long_name="Number of spectral averages (not accounting for overlapping FFTs)",
        units="1",
    ),
    "rg0": MetaData(long_name="Number of lowest range gates", units="1"),
    "prf": MetaData(
        long_name="Pulse Repetition Frequency",
        units="Hz",
    ),
}
