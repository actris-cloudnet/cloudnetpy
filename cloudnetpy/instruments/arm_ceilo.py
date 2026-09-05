"""Module for reading ARM Vaisala ceilometer data."""

import datetime
import logging
from os import PathLike
from uuid import UUID

import netCDF4
import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.arm_utils import read_geolocation
from cloudnetpy.instruments.ceilo import ATTRIBUTES
from cloudnetpy.instruments.ceilometer import Ceilometer, NoiseParam

# ARM unit 1/(sr*km*10000) -> sr-1 m-1
BACKSCATTER_SCALE = 1e-7
# Noise parameters as in the raw Vaisala readers
NOISE_PARAMS = {
    "CT25K": NoiseParam(noise_min=0.7e-7, noise_smooth_min=1.2e-8),
    "CL31": NoiseParam(noise_min=3.1e-8, noise_smooth_min=1.1e-8),
    "CL51": NoiseParam(noise_min=3.1e-8, noise_smooth_min=1.1e-8),
}


def armceilo2nc(
    raw_file: str | PathLike,
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts ARM Vaisala ceilometer data (ceil) into Cloudnet Level 1b
    netCDF file.

    The ARM `ceil.b1` datastream contains the Vaisala CT25K / CL31 / CL51
    messages ingested into netCDF. The data are processed like the raw Vaisala
    files.

    Args:
        raw_file: Daily ARM `ceil` netCDF file, e.g.
            `sgpceilC1.b1.20220601.000000.nc`.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            value pair is `name`. Optional are `latitude`, `longitude` and
            `altitude` (taken from the raw file if missing) and
            `calibration_factor` (default = 1).
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Examples:
          >>> from cloudnetpy.instruments import armceilo2nc
          >>> site_meta = {'name': 'Southern Great Plains'}
          >>> armceilo2nc('sgpceilC1.b1.20220601.000000.nc', 'ceilo.nc', site_meta)

    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = utils.get_uuid(uuid)
    snr_limit = 5
    ceilo = ArmCeilo(raw_file, site_meta, date)
    ceilo.read_file()
    ceilo.check_beta_raw_shape()
    n_negatives = 5 if ceilo.instrument is instruments.CL51 else 20
    ceilo.data["beta"] = ceilo.calc_screened_product(
        ceilo.data["beta_raw"], snr_limit, n_negatives=n_negatives
    )
    ceilo.data["beta_smooth"] = ceilo.calc_beta_smooth(
        ceilo.data["beta"], snr_limit, n_negatives=n_negatives
    )
    # Screen using the smoothed mask like with raw Vaisala files
    mask = ceilo.data["beta_smooth"].mask
    ceilo.data["beta"] = ma.masked_where(mask, ceilo.data["beta_raw"])
    ceilo.data["beta"][ceilo.data["beta"] <= 0] = ma.masked
    ceilo.screen_invalid_values()
    ceilo.screen_sunbeam()
    ceilo.prepare_data()
    ceilo.data_to_cloudnet_arrays()
    ceilo.add_site_geolocation()
    attributes = output.add_time_attribute(ATTRIBUTES, ceilo.date)
    output.update_attributes(ceilo.data, attributes)
    for key in ("beta", "beta_smooth"):
        ceilo.add_snr_info(key, snr_limit)
    output.save_level1b(ceilo, output_file, uuid)
    return uuid


class ArmCeilo(Ceilometer):
    """Class for ARM Vaisala CT25K / CL31 / CL51 ceilometer data."""

    def __init__(
        self,
        full_path: str | PathLike,
        site_meta: dict,
        expected_date: datetime.date | None = None,
    ) -> None:
        super().__init__()
        self.full_path = full_path
        self.site_meta = {**site_meta}
        self.expected_date = expected_date

    def read_file(self) -> None:
        with netCDF4.Dataset(self.full_path) as nc:
            self._set_instrument(nc)
            self._add_geolocation_from_file(nc)
            self.data["range"] = np.array(nc["range"][:], dtype=float)
            self.data["time"] = np.array(nc["time"][:])
            calibration_factor = float(self.site_meta.get("calibration_factor", 1))
            backscatter = ma.masked_invalid(nc["backscatter"][:])
            # Missing profiles are filled with int32 min without _FillValue
            backscatter = ma.masked_less(backscatter, -1e6)
            self.data["beta_raw"] = backscatter * BACKSCATTER_SCALE * calibration_factor
            self.data["calibration_factor"] = calibration_factor
            self.data["zenith_angle"] = float(ma.median(nc["tilt_angle"][:]))
            epoch = utils.get_epoch(nc["time"].units)
        self.get_date_and_time(epoch)

    def _set_instrument(self, nc: netCDF4.Dataset) -> None:
        name = read_model(nc)
        self.instrument = getattr(instruments, name)
        self.noise_param = NOISE_PARAMS[name]

    def _add_geolocation_from_file(self, nc: netCDF4.Dataset) -> None:
        self.site_meta = read_geolocation(nc, self.site_meta)


def read_model(nc: netCDF4.Dataset) -> str:
    """Returns the ceilometer model (CT25K, CL31 or CL51) of an ARM ceil file."""
    model = getattr(nc, "ceilometer_model", "").upper()
    for name in NOISE_PARAMS:
        if name in model:
            return name
    logging.warning("Unknown ceilometer model '%s', assuming CL31", model)
    return "CL31"
