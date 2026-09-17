import datetime
import tempfile
from collections.abc import Sequence
from os import PathLike
from tempfile import TemporaryDirectory
from uuid import UUID

from cloudnetpy import concat_lib, output, utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.instruments.instruments import XPOL
from cloudnetpy.instruments.nc_radar import NcRadar


def xpol2nc(
    input_files: str | PathLike | Sequence[str | PathLike],
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts ProSensing XPOL weather radar data into Cloudnet Level 1b netCDF file.

    This function reads py-ART processed XPOL netCDF files, processes the variables
    and writes a Cloudnet Level 1b netCDF file. Multiple input files can be
    concatenated.

    Args:
        input_files: Path to XPOL netCDF file or list of files to concatenate.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required
            keys are `name`, `latitude`, `longitude` and `altitude`.
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD.

    Returns:
        UUID of the generated file.
    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = utils.get_uuid(uuid)

    keymap = {
        "Zh": "Zh",
        "Zdr": "zdr",
        "Rhohv": "rho_hv",
        "RVel": "v",
        "Sw": "width",
        # "Psidp": "phi_dp",  # TODO: deg2rad?
        # "elevation": "elevation",
        # "azimuth": "azimuth_angle",
    }

    with TemporaryDirectory() as temp_dir:
        if isinstance(input_files, Sequence):
            with tempfile.NamedTemporaryFile(
                dir=temp_dir,
                suffix=".nc",
                delete=False,
            ) as temp_file:
                nc_filename = temp_file.name
                concat_lib.concatenate_files(
                    input_files,
                    nc_filename,
                    variables=list(keymap.keys()),
                    ignore=["altitude"],  # units missing
                )
        else:
            nc_filename = input_files

        with Xpol(nc_filename, site_meta) as xpol:
            xpol.init_data(keymap)
            xpol.date = date
            xpol.add_time_and_range()
            xpol.sort_timestamps()
            xpol.remove_duplicate_timestamps()
            xpol.add_radar_specific_variables()
            xpol.add_site_geolocation()
            # valid_indices = xpol.add_zenith_and_azimuth_angles(
            #     elevation_threshold=1.1,
            #     elevation_diff_threshold=0.1,
            #     azimuth_diff_threshold=0.1,
            # )
            # xpol.screen_time_indices(valid_indices)
            xpol.add_height()
            attributes = output.add_time_attribute({}, xpol.date)
            output.update_attributes(xpol.data, attributes)
            output.save_level1b(xpol, output_file, uuid)
            return uuid


class Xpol(NcRadar):
    def __init__(self, full_path: str | PathLike, site_meta: dict) -> None:
        super().__init__(full_path, site_meta)
        self.instrument = XPOL

    def add_radar_specific_variables(self):
        if not hasattr(self.dataset, "RadarFreq_unit") or not hasattr(
            self.dataset, "RadarFreq_value"
        ):
            msg = "Radar frequency missing"
            raise ValueError(msg)
        if self.dataset.RadarFreq_unit != "GigaHertz":
            msg = "Invalid radar frequency units"
            raise ValueError(msg)
        self.data["radar_frequency"] = CloudnetArray(
            self.dataset.RadarFreq_value, "radar_frequency"
        )

        if not hasattr(self.dataset, "NyquistVelocity_unit") or not hasattr(
            self.dataset, "NyquistVelocity_value"
        ):
            msg = "Nyquist velocity missing"
            raise ValueError(msg)
        if self.dataset.NyquistVelocity_unit != "MetersPerSecond":
            msg = "Invalid nyquist velocity units"
            raise ValueError(msg)
        self.data["nyquist_velocity"] = CloudnetArray(
            self.dataset.NyquistVelocity_value, "nyquist_velocity"
        )
