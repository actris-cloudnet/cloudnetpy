"""Module for reading raw cloud radar data."""

import logging
import os
from collections import OrderedDict
from tempfile import NamedTemporaryFile, TemporaryDirectory

from numpy import ma

from cloudnetpy import concat_lib, output, utils
from cloudnetpy.instruments.instruments import MIRA10, MIRA35
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
        input_filename, keymap = _parse_input_files(raw_mira, temp_dir)

        with Mira(input_filename, site_meta) as mira:
            mira.init_data(keymap)
            if date is not None:
                mira.screen_by_date(date)
                mira.date = date.split("-")
            mira.sort_timestamps()
            mira.remove_duplicate_timestamps()
            mira.linear_to_db(("Zh", "ldr", "SNR"))
            n_profiles = utils.n_elements(mira.time, 5, "time")
            valid_ind = utils.remove_masked_blocks(mira.data["Zh"][:], limit=n_profiles)
            mira.screen_time_indices(valid_ind)

            if "snr_limit" in site_meta and site_meta["snr_limit"] is not None:
                snr_limit = site_meta["snr_limit"]
            else:
                # Empirical values, should be checked
                snr_limit = -30 if mira.instrument == MIRA10 else -17

            # Old MIRA files don't have angle variables.
            if "elevation" not in mira.data:
                mira.append_data(ma.masked_all_like(mira.time.data), "elevation")
            if "azimuth_angle" not in mira.data:
                mira.append_data(ma.masked_all_like(mira.time.data), "azimuth_angle")

            mira.screen_by_snr(snr_limit)
            mira.screen_invalid_ldr()
            mira.mask_invalid_data()
            mira.mask_bad_angles()
            mira.add_time_and_range()
            mira.add_site_geolocation()
            mira.add_radar_specific_variables()
            valid_indices = mira.add_zenith_and_azimuth_angles(
                elevation_threshold=1.1,
                elevation_diff_threshold=1e-6,
                azimuth_diff_threshold=1e-3,
                zenith_offset=site_meta.get("zenith_offset"),
                azimuth_offset=site_meta.get("azimuth_offset"),
            )
            mira.screen_time_indices(valid_indices)
            mira.add_height()
            mira.test_if_all_masked()
        attributes = output.add_time_attribute(ATTRIBUTES, mira.date)
        output.update_attributes(mira.data, attributes)
        return output.save_level1b(mira, output_file, uuid)


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
        if "model" not in site_meta or site_meta["model"] == "mira-35":
            self.instrument = MIRA35
        elif site_meta["model"] == "mira-10":
            self.instrument = MIRA10
        else:
            msg = f"Invalid model: {site_meta['model']}"
            raise ValueError(msg)

    def screen_by_date(self, expected_date: str) -> None:
        """Screens incorrect time stamps."""
        time_stamps = self.getvar("time")
        valid_indices = []
        for ind, timestamp in enumerate(time_stamps):
            if not timestamp:
                continue
            date = "-".join(utils.seconds2date(timestamp, self.epoch)[:3])
            if date == expected_date:
                valid_indices.append(ind)
        self.screen_time_indices(valid_indices)

    def _init_mira_date(self) -> list[str]:
        time_stamps = self.getvar("time")
        return utils.seconds2date(float(time_stamps[0]), self.epoch)[:3]

    def screen_invalid_ldr(self) -> None:
        """Masks LDR in MIRA STSR mode data.
        Is there a better way to identify this mode?
        """
        if "ldr" not in self.data:
            return
        ldr = self.data["ldr"][:]
        if ma.mean(ldr) > 0:
            logging.warning(
                "LDR values suspiciously high. Mira in STSR mode? "
                "Screening all LDR for now.",
            )
            self.data["ldr"].data[:] = ma.masked

    def mask_bad_angles(self) -> None:
        """Masks clearly bad elevation and azimuth angles."""
        limits = {
            "elevation": (0, 180),
            "azimuth_angle": (-360, 360),
        }
        for key, (lower, upper) in limits.items():
            if (array := self.data[key].data) is not None:
                margin = (upper - lower) * 0.05
                array[array < (lower - margin)] = ma.masked
                array[array > (upper + margin)] = ma.masked


def _parse_input_files(input_files: str | list[str], temp_dir: str) -> tuple:
    if isinstance(input_files, list) or os.path.isdir(input_files):
        with NamedTemporaryFile(
            dir=temp_dir,
            suffix=".nc",
            delete=False,
        ) as temp_file:
            input_filename = temp_file.name
            if isinstance(input_files, list):
                valid_files = sorted(input_files)
            else:
                valid_files = utils.get_sorted_filenames(input_files, ".znc")
                if not valid_files:
                    valid_files = utils.get_sorted_filenames(input_files, ".mmclx")

            if not valid_files:
                msg = (
                    (
                        f"Neither znc nor mmclx files found {input_files}. "
                        f"Please check your input."
                    ),
                )
                raise FileNotFoundError(msg)

            filetypes = list({f.split(".")[-1].lower() for f in valid_files})

            if len(filetypes) > 1:
                err_msg = "Mixed mmclx and znc files. Please use only one filetype."
                raise TypeError(err_msg)

            keymap = _get_keymap(filetypes[0])

            variables = list(keymap.keys())
            concat_lib.concatenate_files(
                valid_files,
                input_filename,
                variables=variables,
                ignore=_get_ignored_variables(filetypes[0]),
            )
    else:
        input_filename = input_files
        keymap = _get_keymap(input_filename.split(".")[-1])

    return input_filename, keymap


def _get_ignored_variables(filetype: str) -> list | None:
    """Returns variables to ignore for METEK MIRA-35 cloud radar concat."""
    _check_file_type(filetype)
    # Ignore spectral variables for now
    keymaps = {
        "znc": ["DropSize", "SPCco", "SPCcx", "SPCcocxRe", "SPCcocxIm", "doppler"],
        "mmclx": None,
    }

    return keymaps.get(filetype.lower(), keymaps.get("mmclx"))


def _get_keymap(filetype: str) -> dict:
    """Returns a dictionary mapping the variables in the raw data to the processed
    Cloudnet file.
    """
    _check_file_type(filetype)

    # Order is relevant with the new znc files from STSR radar
    keymaps = {
        "znc": OrderedDict(
            [
                ("Zg", "Zh"),  # fallback
                ("Zh2l", "Zh"),
                ("VELg", "v"),  # fallback
                ("VELh2l", "v"),
                ("RMSg", "width"),  # fallback
                ("RMSh2l", "width"),
                ("LDRg", "ldr"),  # fallback
                ("LDRh2l", "ldr"),
                ("SNRg", "SNR"),  # fallback
                ("SNRh2l", "SNR"),
                ("elv", "elevation"),
                ("azi", "azimuth_angle"),
                ("nfft", "nfft"),
                ("nave", "nave"),
                ("prf", "prf"),
                ("rg0", "rg0"),
            ],
        ),
        "mmclx": OrderedDict(
            [
                ("Ze", "Zh"),  # fallback for old mmclx files
                ("Zg", "Zh"),
                ("VELg", "v"),
                ("RMSg", "width"),
                ("LDRg", "ldr"),
                ("SNRg", "SNR"),
                ("elv", "elevation"),
                ("azi", "azimuth_angle"),
                ("nfft", "nfft"),
                ("nave", "nave"),
                ("prf", "prf"),
                ("rg0", "rg0"),
                ("NyquistVelocity", "NyquistVelocity"),  # variable in some mmclx files
            ]
        ),
    }

    return keymaps.get(filetype.lower(), keymaps["mmclx"])


def _check_file_type(filetype: str) -> None:
    known_filetypes = ["znc", "mmclx"]
    if filetype.lower() not in known_filetypes:
        msg = f"Filetype must be one of {known_filetypes}"
        raise ValueError(msg)


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
    "zenith_offset": MetaData(
        long_name="Zenith offset of the instrument",
        units="degrees",
        comment="Zenith offset applied.",
    ),
    "azimuth_offset": MetaData(
        long_name="Azimuth offset of the instrument (positive clockwise from north)",
        units="degrees",
        comment="Azimuth offset applied.",
    ),
}
