"""Module for reading raw cloud radar data."""
import os
from collections import OrderedDict
from tempfile import TemporaryDirectory

from cloudnetpy import concat_lib, output, utils
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments.instruments import MIRA35
from cloudnetpy.instruments.nc_radar import NcRadar
from cloudnetpy.metadata import MetaData


def mira2nc(
    raw_mira: str | list[str],
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | None = None,
) -> str:
    """Converts METEK MIRA-35 cloud radar data into Cloudnet Level 1b netCDF file.

    This function converts raw MIRA file(s) into a much smaller file that
    contains only the relevant data and can be used in further processing
    steps.

    Args:
        raw_mira: Filename of a daily MIRA .mmclx or .zncfile. Can be also a folder
            containing several non-concatenated .mmclx or .znc files from one day
            or list of files. znc files take precedence because they are the newer
            filetype
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            value pair is `name`.
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.
        FileNotFoundError: No suitable input files found.
        ValueError: Wrong suffix in input file(s).
        TypeError: Mixed mmclx and znc files.

    Examples:
          >>> from cloudnetpy.instruments import mira2nc
          >>> site_meta = {'name': 'Vehmasmaki'}
          >>> mira2nc('raw_radar.mmclx', 'radar.nc', site_meta)
          >>> mira2nc('raw_radar.znc', 'radar.nc', site_meta)
          >>> mira2nc('/one/day/of/mira/mmclx/files/', 'radar.nc', site_meta)
          >>> mira2nc('/one/day/of/mira/znc/files/', 'radar.nc', site_meta)

    """

    with TemporaryDirectory() as temp_dir:
        if isinstance(raw_mira, list) or os.path.isdir(raw_mira):
            # better naming would be concat_filename but to be directly
            # compatible with the opening of the output we stick to input_filename
            input_filename = f"{temp_dir}/tmp.nc"
            # passed in is a list of files
            if isinstance(raw_mira, list):
                valid_files = sorted(raw_mira)
            else:
                # passed in is a path with potentially files
                valid_files = utils.get_sorted_filenames(raw_mira, ".znc")
                if not valid_files:
                    valid_files = utils.get_sorted_filenames(raw_mira, ".mmclx")

            if not valid_files:
                raise FileNotFoundError(
                    "Neither znc nor mmclx files found "
                    + f"{raw_mira}. Please check your input."
                )

            valid_files = utils.get_files_with_common_range(valid_files)

            # get unique filetypes
            filetypes = list({f.split(".")[-1].lower() for f in valid_files})

            if len(filetypes) > 1:
                raise TypeError(
                    "mira2nc only supports a singlefile type as input",
                    "either mmclx or znc",
                )

            keymap = _mirakeymap(filetypes[0])

            variables = list(keymap.keys())
            concat_lib.concatenate_files(
                valid_files,
                input_filename,
                variables=variables,
                ignore=_miraignorevar(filetypes[0]),
                allow_difference=["nave", "ovl"],
            )
        else:
            input_filename = raw_mira
            keymap = _mirakeymap(input_filename.split(".")[-1])

        with Mira(input_filename, site_meta) as mira:
            mira.init_data(keymap)
            if date is not None:
                mira.screen_by_date(date)
                mira.date = date.split("-")
            mira.sort_timestamps()
            mira.remove_duplicate_timestamps()
            mira.linear_to_db(("Zh", "ldr", "SNR"))
            mira.screen_by_snr()
            mira.mask_invalid_data()
            mira.add_time_and_range()
            mira.add_site_geolocation()
            mira.add_radar_specific_variables()
            valid_indices = mira.add_zenith_and_azimuth_angles()
            mira.screen_time_indices(valid_indices)
            mira.add_height()
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
        self.screen_time_indices(valid_indices)

    def _init_mira_date(self) -> list[str]:
        time_stamps = self.getvar("time")
        return utils.seconds2date(time_stamps[0], self.epoch)[:3]


def _miraignorevar(filetype: str) -> list | None:
    """Returns the vars to ignore for METEK MIRA-35 cloud radar concat.

    This function return the nc variable names that should be ignored when
    concatenating several files, a requirement needed when a path/list of files
    can be passed in to mira2nc, at the moment (08.2023) only relevant for znc.

    Args:
        filetype: Either znc or mmclx

    Returns:
        Appropriate list of variables to ignore for the file type

    Raises:
        TypeError: Not a valid filetype given, must be string.
        ValueError: Not a known filetype given, must be znc or mmclx

    Examples:
        not meant to be called directly by user

    """
    known_filetypes = ["znc", "mmclx"]
    if not isinstance(filetype, str):
        raise TypeError("Filetype must be string")

    if filetype.lower() not in known_filetypes:
        raise ValueError(f"Filetype must be one of {known_filetypes}")

    keymaps = {"znc": ["DropSize"], "mmclx": None}

    return keymaps.get(filetype.lower(), keymaps.get("mmclx"))


def _mirakeymap(filetype: str) -> dict:
    """Returns the ncvariables to cloudnetpy mapping for METEK MIRA-35 cloud radar.

    This function return the appropriate keymap (even for STSR polarimetric
    config) for cloudnetpy to take the appropriate variables from the netCDF
    whether mmclx (old format) or znc (new format).

    Args:
        filetype: Either znc or mmclx

    Returns:
        Appropriate keymap for the file type

    Raises:
        TypeError: Not a valid filetype given, must be string.
        ValueError: Not a known filetype given, must be znc or mmclx

    Examples:
          not meant to be called directly by user

    """
    known_filetypes = ["znc", "mmclx"]
    if not isinstance(filetype, str):
        raise TypeError("Filetype must be string")

    if filetype.lower() not in known_filetypes:
        raise ValueError(f"Filetype must be one of {known_filetypes}")

    # ordered dict here because that way the order is kept, which means
    # we will get Zh2l over as Zh over Zg, which is relevant for the new
    # znc files of an STSR radar
    keymaps = {
        "znc": OrderedDict(
            [
                ("Zg", "Zh"),
                ("Zh2l", "Zh"),
                ("VELg", "v"),
                ("VELh2l", "v"),
                ("RMSg", "width"),
                ("RMSh2l", "width"),
                ("LDRg", "ldr"),
                ("LDRh2l", "ldr"),
                ("SNRg", "SNR"),
                ("SNRh2l", "SNR"),
                ("elv", "elevation"),
                ("azi", "azimuth_angle"),
                ("aziv", "azimuth_velocity"),
                ("nfft", "nfft"),
                ("nave", "nave"),
                ("prf", "prf"),
                ("rg0", "rg0"),
            ]
        ),
        "mmclx": {
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
        },
    }

    return keymaps.get(filetype.lower(), keymaps["mmclx"])


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
