"""Module for reading raw cloud radar data."""

import datetime
import os
import tempfile
from os import PathLike
from tempfile import TemporaryDirectory
from uuid import UUID

import numpy as np

from cloudnetpy import concat_lib, output, utils
from cloudnetpy.instruments.instruments import COPERNICUS
from cloudnetpy.instruments.nc_radar import ChilboltonRadar
from cloudnetpy.metadata import COMMON_ATTRIBUTES, MetaData


def copernicus2nc(
    raw_files: str | PathLike,
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
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
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = utils.get_uuid(uuid)

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

    nc_filename: str | PathLike
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
            copernicus.fix_range_offset(site_meta)
            copernicus.screen_negative_ranges()
            copernicus.add_radar_specific_variables()
            copernicus.add_nyquist_velocity(keymap)
            copernicus.add_site_geolocation()
            valid_indices = copernicus.add_zenith_and_azimuth_angles(
                elevation_threshold=1.1,
                elevation_diff_threshold=0.5,
                azimuth_diff_threshold=0.1,
            )
            copernicus.screen_time_indices(valid_indices)
            copernicus.add_height()
            copernicus.test_if_all_masked()
        attributes = output.add_time_attribute(ATTRIBUTES, copernicus.date)
        output.update_attributes(copernicus.data, attributes)
        output.save_level1b(copernicus, output_file, uuid)
        return uuid


class Copernicus(ChilboltonRadar):
    """Class for Copernicus raw radar data. Child of ChilboltonRadar().

    Args:
        full_path: Filename of a daily Copernicus .nc NetCDF file.
        site_meta: Site properties in a dictionary. Required keys are: `name`.

    """

    def __init__(self, full_path: str | PathLike, site_meta: dict) -> None:
        super().__init__(full_path, site_meta)
        self.instrument = COPERNICUS

    def calibrate_reflectivity(self) -> None:
        zed_hc = self.dataset["ZED_HC"]
        offset_applied = getattr(zed_hc, "applied_calibration_offset", 0)

        # Estimated by comparing with MIRA-35 data:
        default_offset = -149.5
        zh_offset = self.site_meta.get("Zh_offset", default_offset)

        self.data["Zh"].data[:] = self.data["Zh"].data[:] - offset_applied + zh_offset
        self.append_data(np.array(zh_offset, dtype=np.float32), "Zh_offset")

    def fix_range_offset(self, site_meta: dict) -> None:
        range_var = self.dataset["range"]
        offset_applied = getattr(range_var, "range_offset", 0)

        # Estimated by comparing with MIRA-35 data:
        default_offset = -720
        range_offset = site_meta.get("range_offset", default_offset)

        self.data["range"].data[:] = (
            self.data["range"].data[:] - offset_applied + range_offset
        )
        self.append_data(np.array(range_offset, dtype=float), "range_offset")

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
    "Zh_offset": MetaData(
        long_name="Radar reflectivity calibration offset",
        units="dBZ",
        comment=(
            "Calibration offset applied after removing the original offset "
            "from the raw files."
        ),
        dimensions=None,
    ),
    "Zh": COMMON_ATTRIBUTES["Zh"]._replace(ancillary_variables="Zh_offset"),
    "range_offset": MetaData(
        long_name="Radar range offset",
        units="m",
        comment=(
            "Range offset applied after removing the original offset "
            "from the raw files."
        ),
        dimensions=None,
    ),
    "range": COMMON_ATTRIBUTES["range"]._replace(ancillary_variables="range_offset"),
    "antenna_diameter": MetaData(
        long_name="Antenna diameter", units="m", dimensions=("time",)
    ),
    "beamwidthV": MetaData(
        long_name="Vertical angular beamwidth", units="degree", dimensions=("time",)
    ),
    "beamwidthH": MetaData(
        long_name="Horizontal angular beamwidth", units="degree", dimensions=("time",)
    ),
}
