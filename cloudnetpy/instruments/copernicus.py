"""Module for reading raw cloud radar data."""

import os
import tempfile
from tempfile import TemporaryDirectory

import numpy as np

from cloudnetpy import concat_lib, output, utils
from cloudnetpy.instruments.instruments import COPERNICUS
from cloudnetpy.instruments.nc_radar import ChilboltonRadar
from cloudnetpy.metadata import MetaData


def copernicus2nc(
    raw_files: str,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | None = None,
) -> str:
    """Converts 'Copernicus' cloud radar data into Cloudnet Level 1b netCDF file.

    Args:
        raw_files: Input file name or folder containing multiple input files.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            value pair is `name`. Optional are `latitude`, `longitude`, `altitude` and
            'calibration_offset' (default = -146.8).
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    Examples:
          >>> from cloudnetpy.instruments import copernicus2nc
          >>> site_meta = {'name': 'Chilbolton'}
          >>> copernicus2nc('raw_radar.nc', 'radar.nc', site_meta)
          >>> copernicus2nc('/one/day/of/copernicus/files/', 'radar.nc', site_meta)

    """
    keymap = {
        "ZED_HC": "Zh",
        "VEL_HC": "v",
        "SPW_HC": "width",
        "LDR_C": "ldr",
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
            with tempfile.NamedTemporaryFile(
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

        with Copernicus(nc_filename, site_meta) as copernicus:
            copernicus.init_data(keymap)
            copernicus.add_time_and_range()
            if date is not None:
                copernicus.check_date(date)
            copernicus.sort_timestamps()
            copernicus.remove_duplicate_timestamps()
            copernicus.calibrate_reflectivity()
            copernicus.screen_using_top_gates_snr()
            copernicus.mask_corrupted_values()
            copernicus.mask_first_range_gates()
            copernicus.mask_invalid_data()
            copernicus.add_time_and_range()
            copernicus.fix_range_offset(site_meta)
            copernicus.screen_negative_ranges()
            copernicus.add_radar_specific_variables()
            copernicus.add_nyquist_velocity(keymap)
            copernicus.add_site_geolocation()
            valid_indices = copernicus.add_zenith_and_azimuth_angles()
            copernicus.screen_time_indices(valid_indices)
            copernicus.add_height()
        attributes = output.add_time_attribute(ATTRIBUTES, copernicus.date)
        output.update_attributes(copernicus.data, attributes)
        return output.save_level1b(copernicus, output_file, uuid)


class Copernicus(ChilboltonRadar):
    """Class for Copernicus raw radar data. Child of ChilboltonRadar().

    Args:
        full_path: Filename of a daily Copernicus .nc NetCDF file.
        site_meta: Site properties in a dictionary. Required keys are: `name`.

    """

    def __init__(self, full_path: str, site_meta: dict):
        super().__init__(full_path, site_meta)
        self.instrument = COPERNICUS

    def calibrate_reflectivity(self) -> None:
        default_offset = -146.8  # TODO: check this value
        calibration_factor = self.site_meta.get("calibration_offset", default_offset)
        self.data["Zh"].data[:] += calibration_factor
        self.append_data(np.array(calibration_factor), "calibration_offset")

    def mask_corrupted_values(self) -> None:
        """Experimental masking of corrupted Copernicus data.

        Notes:
            This method is based on a few days of test data only. Should be improved
            and tested more carefully in the future.
        """
        thresholds = {"width": 3, "v": 9}
        for key, value in thresholds.items():
            ind = np.where(np.abs(self.data[key][:]) > value)
            self.data["v"].mask_indices(ind)

    def fix_range_offset(self, site_meta: dict) -> None:
        """Fixes range offset."""
        range_offset = site_meta.get("range_offset", 0)
        self.data["range"].data[:] += range_offset
        self.append_data(np.array(range_offset, dtype=float), "range_offset")

    def screen_negative_ranges(self) -> None:
        """Screens negative range values."""
        valid_ind = np.where(self.data["range"][:] >= 0)[0]
        for key, cloudnet_array in self.data.items():
            try:
                data = cloudnet_array[:]
                if data.ndim == 2:
                    cloudnet_array.data = data[:, valid_ind]
                elif key == "range":
                    cloudnet_array.data = data[valid_ind]
            except IndexError:
                continue


ATTRIBUTES = {
    "calibration_offset": MetaData(
        long_name="Radar reflectivity calibration offset",
        units="dBZ",
        comment="Calibration offset applied.",
    ),
    "range_offset": MetaData(
        long_name="Radar range offset",
        units="m",
        comment="Range offset applied.",
    ),
    "antenna_diameter": MetaData(long_name="Antenna diameter", units="m"),
    "beamwidthV": MetaData(long_name="Vertical angular beamwidth", units="degree"),
    "beamwidthH": MetaData(long_name="Horizontal angular beamwidth", units="degree"),
}
