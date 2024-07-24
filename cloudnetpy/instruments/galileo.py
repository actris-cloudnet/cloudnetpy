"""Module for reading raw Galileo cloud radar data."""

import os
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np

from cloudnetpy import concat_lib, output, utils
from cloudnetpy.instruments.instruments import GALILEO
from cloudnetpy.instruments.nc_radar import ChilboltonRadar
from cloudnetpy.metadata import MetaData


def galileo2nc(
    raw_files: str,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | None = None,
) -> str:
    """Converts 'Galileo' cloud radar data into Cloudnet Level 1b netCDF file.

    Args:
        raw_files: Input file name or folder containing multiple input files.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            value pair is `name`. Optional are `latitude`, `longitude`, `altitude` and
            `snr_limit` (default = 3).
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    Examples:
          >>> from cloudnetpy.instruments import galileo2nc
          >>> site_meta = {'name': 'Chilbolton'}
          >>> galileo2nc('raw_radar.nc', 'radar.nc', site_meta)
          >>> galileo2nc('/one/day/of/galileo/files/', 'radar.nc', site_meta)

    """
    keymap = {
        "ZED_HC": "Zh",
        "VEL_HC": "v",
        "SPW_HC": "width",
        "LDR_HC": "ldr",
        "SNR_HC": "SNR",
        "elevation": "elevation",
        "azimuth": "azimuth_angle",
        "height": "altitude",
        "antenna_diameter": "antenna_diameter",
        "beamwidthV": "beamwidthV",
        "beamwidthH": "beamwidthH",
    }

    with TemporaryDirectory() as temp_dir:
        if os.path.isdir(raw_files):
            with NamedTemporaryFile(
                dir=temp_dir,
                suffix=".nc",
                delete=False,
            ) as temp_file:
                nc_filename = temp_file.name
                valid_filenames = utils.get_sorted_filenames(raw_files, ".nc")
                valid_filenames = utils.get_files_with_variables(
                    valid_filenames, ["time", "ZED_HC"]
                )
                valid_filenames = utils.get_files_with_common_range(valid_filenames)
                variables = list(keymap.keys())
                concat_lib.concatenate_files(
                    valid_filenames,
                    nc_filename,
                    variables=variables,
                )
        else:
            nc_filename = raw_files

        with Galileo(nc_filename, site_meta) as galileo:
            galileo.init_data(keymap)
            galileo.add_time_and_range()
            if date is not None:
                galileo.check_date(date)
            galileo.sort_timestamps()
            galileo.remove_duplicate_timestamps()
            snr_limit = site_meta.get("snr_limit", 3)
            galileo.screen_by_snr(snr_limit=snr_limit)
            galileo.mask_clutter()
            galileo.mask_invalid_data()
            galileo.add_time_and_range()
            galileo.add_radar_specific_variables()
            galileo.add_nyquist_velocity(keymap)
            galileo.add_site_geolocation()
            valid_indices = galileo.add_zenith_and_azimuth_angles()
            galileo.screen_time_indices(valid_indices)
            galileo.add_height()
        attributes = output.add_time_attribute(ATTRIBUTES, galileo.date)
        output.update_attributes(galileo.data, attributes)
        return output.save_level1b(galileo, output_file, uuid)


class Galileo(ChilboltonRadar):
    """Class for Galileo raw radar data. Child of ChilboltonRadar().

    Args:
        full_path: Filename of a daily Galileo .nc NetCDF file.
        site_meta: Site properties in a dictionary. Required keys are: `name`.

    """

    def __init__(self, full_path: str, site_meta: dict):
        super().__init__(full_path, site_meta)
        self.date = self._init_date()
        self.instrument = GALILEO

    def mask_clutter(self) -> None:
        """Masks clutter."""
        # Only strong Z values are valid
        n_low_gates = 15
        ind = np.where(self.data["Zh"][:, :n_low_gates] < -15) and np.where(
            self.data["ldr"][:, :n_low_gates] > -5,
        )
        self.data["v"].mask_indices(ind)


ATTRIBUTES = {
    "antenna_diameter": MetaData(long_name="Antenna diameter", units="m"),
    "beamwidthV": MetaData(long_name="Vertical angular beamwidth", units="degree"),
    "beamwidthH": MetaData(long_name="Horizontal angular beamwidth", units="degree"),
}
