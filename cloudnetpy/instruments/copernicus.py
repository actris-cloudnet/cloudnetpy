"""Module for reading raw cloud radar data."""
import os
from tempfile import TemporaryDirectory
from typing import List, Optional

import numpy as np

from cloudnetpy import concat_lib, output, utils
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import general
from cloudnetpy.instruments.instruments import COPERNICUS
from cloudnetpy.instruments.nc_radar import NcRadar
from cloudnetpy.metadata import MetaData


def copernicus2nc(
    raw_files: str,
    output_file: str,
    site_meta: dict,
    uuid: Optional[str] = None,
    date: Optional[str] = None,
) -> str:
    """Converts 'Copernicus' cloud radar data into Cloudnet Level 1b netCDF file.

    Args:
        raw_files: Input file name or folder containing multiple input files.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key value pair
            is `name`. Optional is 'calibration_offset'.
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
            nc_filename = f"{temp_dir}/tmp.nc"
            valid_filenames = utils.get_sorted_filenames(raw_files, ".nc")
            valid_filenames = general.get_files_with_common_range(valid_filenames)
            variables = list(keymap.keys())
            concat_lib.concatenate_files(valid_filenames, nc_filename, variables=variables)
        else:
            nc_filename = raw_files

        with Copernicus(nc_filename, site_meta) as copernicus:
            copernicus.init_data(keymap)
            if date is not None:
                copernicus.check_date(date)
            copernicus.sort_timestamps()
            copernicus.remove_duplicate_timestamps()
            copernicus.calibrate_reflectivity()
            copernicus.screen_by_snr(snr_limit=3)
            copernicus.mask_corrupted_values()
            copernicus.mask_invalid_data()
            copernicus.add_time_and_range()
            general.add_radar_specific_variables(copernicus)
            copernicus.add_nyquist_velocity(keymap)
            general.add_site_geolocation(copernicus)
            valid_indices = copernicus.add_zenith_and_azimuth_angles()
            general.screen_time_indices(copernicus, valid_indices)
            general.add_height(copernicus)
        attributes = output.add_time_attribute(ATTRIBUTES, copernicus.date)
        output.update_attributes(copernicus.data, attributes)
        uuid = output.save_level1b(copernicus, output_file, uuid)
        return uuid


class Copernicus(NcRadar):
    """Class for Copernicus raw radar data. Child of NcRadar().

    Args:
        full_path: Filename of a daily Copernicus .nc NetCDF file.
        site_meta: Site properties in a dictionary. Required keys are: `name`.

    """

    def __init__(self, full_path: str, site_meta: dict):
        super().__init__(full_path, site_meta)
        self.date = self._init_date()
        self.instrument = COPERNICUS

    def check_date(self, date: str):
        if self.date != date.split("-"):
            raise ValidTimeStampError

    def calibrate_reflectivity(self):
        default_offset = -146.8  # TODO: check this value
        calibration_factor = self.site_meta.get("calibration_offset", default_offset)
        self.data["Zh"].data[:] += calibration_factor
        self.append_data(np.array(calibration_factor), "calibration_offset")

    def mask_corrupted_values(self):
        """Experimental masking of corrupted Copernicus data.

        Notes:
            This method is based on a few days of test data only. Should be improved and tested
            more carefully in the future.
        """
        thresholds = {"width": 3, "v": 9}
        for key, value in thresholds.items():
            ind = np.where(np.abs(self.data[key][:]) > value)
            self.data["v"].mask_indices(ind)

    def add_nyquist_velocity(self, keymap: dict):
        key = [key for key, value in keymap.items() if value == "v"][0]
        folding_velocity = self.dataset.variables[key].folding_velocity
        self.append_data(np.array(folding_velocity), "nyquist_velocity")

    def _init_date(self) -> List[str]:
        epoch = utils.get_epoch(self.dataset["time"].units)
        return [str(x).zfill(2) for x in epoch]


ATTRIBUTES = {
    "calibration_offset": MetaData(
        long_name="Radar reflectivity calibration offset",
        units="1",
        comment="Calibration offset applied.",
    ),
    "antenna_diameter": MetaData(long_name="Antenna diameter", units="m"),
    "beamwidthV": MetaData(long_name="Vertical angular beamwidth", units="degree"),
    "beamwidthH": MetaData(long_name="Horizontal angular beamwidth", units="degree"),
}
