from os import PathLike

import numpy as np

from cloudnetpy import output
from cloudnetpy.constants import G_TO_KG, MM_H_TO_M_S
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments.instruments import FMCW94
from cloudnetpy.instruments.nc_radar import NcRadar
from cloudnetpy.metadata import MetaData


def bowtie2nc(
    bowtie_file: str | PathLike,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | None = None,
) -> str:
    """Converts data from 'BOW-TIE' campaign cloud radar on RV-Meteor into
       Cloudnet Level 1b netCDF file.

    Args:
        bowtie_file: Input filename.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            value pair is `name`. Optional are `latitude`, `longitude`, `altitude`.
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    """
    keymap = {
        "Zh": "Zh",
        "v": "v",
        "width": "width",
        "ldr": "ldr",
        "kurt": "kurtosis",
        "Skew": "skewness",
        "SNR": "SNR",
        "time": "time",
        "range": "range",
        "lwp": "lwp",
        "SurfRelHum": "relative_humidity",
        "rain": "rainfall_rate",
        "Nyquist_velocity": "nyquist_velocity",
        "range_offsets": "chirp_start_indices",
    }

    with Bowtie(bowtie_file, site_meta) as bowtie:
        bowtie.init_data(keymap)
        bowtie.add_time_and_range()
        if date is not None:
            bowtie.check_date(date)
        bowtie.add_radar_specific_variables()
        bowtie.add_site_geolocation()
        bowtie.add_height()
        bowtie.convert_units()
        bowtie.fix_chirp_start_indices()
        bowtie.test_if_all_masked()
    attributes = output.add_time_attribute(ATTRIBUTES, bowtie.date)
    output.update_attributes(bowtie.data, attributes)
    return output.save_level1b(bowtie, output_file, uuid)


class Bowtie(NcRadar):
    def __init__(self, full_path: str | PathLike, site_meta: dict):
        super().__init__(full_path, site_meta)
        self.instrument = FMCW94
        self.date = self.get_date()

    def convert_units(self):
        self.data["lwp"].data *= G_TO_KG
        self.data["rainfall_rate"].data *= MM_H_TO_M_S
        self.data["relative_humidity"].data /= 100

    def fix_chirp_start_indices(self) -> None:
        array = self.data["chirp_start_indices"].data
        self.data["chirp_start_indices"].data = np.array(array, dtype=np.int32)
        self.data["chirp_start_indices"].data_type = "int32"

    def check_date(self, date: str):
        if "-".join(self.date) != date:
            raise ValidTimeStampError


ATTRIBUTES: dict = {
    "v": MetaData(
        long_name="Doppler velocity",
        units="m s-1",
        comment=(
            "This parameter is the radial component of the velocity, with positive\n"
            "velocities are away from the radar. It was corrected for the heave\n"
            "motion of the ship. A rolling average over 3 time steps has been\n"
            "applied to it."
        ),
    ),
}
