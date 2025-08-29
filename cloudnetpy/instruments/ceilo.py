"""Module for reading and processing Vaisala / Lufft ceilometers."""

import datetime
import logging
import os.path
from itertools import islice
from os import PathLike
from uuid import UUID

import netCDF4
import numpy as np
from ceilopyter import read_ct25k
from numpy import ma

from cloudnetpy import output
from cloudnetpy.instruments.cl61d import Cl61d
from cloudnetpy.instruments.lufft import LufftCeilo
from cloudnetpy.instruments.vaisala import ClCeilo, Cs135, Ct25k
from cloudnetpy.metadata import COMMON_ATTRIBUTES, MetaData
from cloudnetpy.utils import get_uuid


def ceilo2nc(
    full_path: str | PathLike,
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts Vaisala, Lufft and Campbell Scientific ceilometer data into
    Cloudnet Level 1b netCDF file.

    This function reads raw Vaisala (CT25k, CL31, CL51, CL61), Lufft
    (CHM 15k, CHM 15k-x) and Campbell Scientific (CS135) ceilometer files and writes
    the data into netCDF file. Three variants of the backscatter are saved:

        1. Raw backscatter, `beta_raw`
        2. Signal-to-noise screened backscatter, `beta`
        3. SNR-screened backscatter with smoothed weak background, `beta_smooth`

    With CL61 two additional depolarisation parameters are saved:

        1. Signal-to-noise screened depolarisation, `depolarisation`
        2. SNR-screened depolarisation with smoothed weak background,
           `depolarisation_smooth`

    CL61 screened backscatter is screened using beta_smooth mask to improve detection
    of weak aerosol layers and supercooled liquid clouds.

    Args:
        full_path: Ceilometer file name.
        output_file: Output file name, e.g. 'ceilo.nc'.
        site_meta: Dictionary containing information about the site and instrument.
            Required key value pairs are `name` and `altitude` (metres above mean
            sea level). Also, 'calibration_factor' is recommended because the default
            value is probably incorrect. If the background noise is *not*
            range-corrected, you must define: {'range_corrected': False}.
            You can also explicitly set the instrument model with
            e.g. {'model': 'cl61d'}.
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        RuntimeError: Failed to read or process raw ceilometer data.

    Examples:
        >>> from cloudnetpy.instruments import ceilo2nc
        >>> site_meta = {'name': 'Mace-Head', 'altitude': 5}
        >>> ceilo2nc('vaisala_raw.txt', 'vaisala.nc', site_meta)
        >>> site_meta = {'name': 'Juelich', 'altitude': 108,
        'calibration_factor': 2.3e-12}
        >>> ceilo2nc('chm15k_raw.nc', 'chm15k.nc', site_meta)

    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = get_uuid(uuid)
    snr_limit = 5
    ceilo_obj = _initialize_ceilo(full_path, site_meta, date)
    calibration_factor = site_meta.get("calibration_factor")
    range_corrected = site_meta.get("range_corrected", True)
    if range_corrected is False:
        logging.warning("Raw data not range-corrected.")
    if isinstance(ceilo_obj, Ct25k):
        c_obj = read_ct25k(full_path, calibration_factor, range_corrected)
        ceilo_obj.data["beta"] = c_obj.beta
        ceilo_obj.data["beta_raw"] = c_obj.beta_raw
        ceilo_obj.data["time"] = c_obj.time
        ceilo_obj.data["range"] = c_obj.range
        if c_obj.zenith_angle is not None:
            ceilo_obj.data["zenith_angle"] = np.median(c_obj.zenith_angle)
        ceilo_obj.data["calibration_factor"] = c_obj.calibration_factor
        ceilo_obj.sort_time()
        ceilo_obj.screen_date()
        ceilo_obj.convert_to_fraction_hour()
    else:
        ceilo_obj.read_ceilometer_file(calibration_factor)
        ceilo_obj.check_beta_raw_shape()
        n_negatives = _get_n_negatives(ceilo_obj)
        ceilo_obj.data["beta"] = ceilo_obj.calc_screened_product(
            ceilo_obj.data["beta_raw"],
            snr_limit,
            range_corrected=range_corrected,
            n_negatives=n_negatives,
        )
        ceilo_obj.data["beta_smooth"] = ceilo_obj.calc_beta_smooth(
            ceilo_obj.data["beta"],
            snr_limit,
            range_corrected=range_corrected,
            n_negatives=n_negatives,
        )
        if ceilo_obj.instrument is None or ceilo_obj.instrument.model is None:
            msg = "Failed to read ceilometer model"
            raise RuntimeError(msg)
        if (
            any(
                model in ceilo_obj.instrument.model.lower()
                for model in ("cl61", "chm15k", "chm15kx", "cl51", "cl31")
            )
            and range_corrected
        ):
            mask = ceilo_obj.data["beta_smooth"].mask
            ceilo_obj.data["beta"] = ma.masked_where(mask, ceilo_obj.data["beta_raw"])
            ceilo_obj.data["beta"][ceilo_obj.data["beta"] <= 0] = ma.masked
            if "depolarisation" in ceilo_obj.data:
                ceilo_obj.data["depolarisation"].mask = ceilo_obj.data["beta"].mask
    ceilo_obj.screen_depol()
    ceilo_obj.screen_invalid_values()
    ceilo_obj.screen_sunbeam()
    ceilo_obj.prepare_data()
    ceilo_obj.data_to_cloudnet_arrays()
    ceilo_obj.add_site_geolocation()
    attributes = output.add_time_attribute(ATTRIBUTES, ceilo_obj.date)
    output.update_attributes(ceilo_obj.data, attributes)
    for key in ("beta", "beta_smooth"):
        ceilo_obj.add_snr_info(key, snr_limit)
    output.save_level1b(ceilo_obj, output_file, uuid)
    return uuid


def _get_n_negatives(ceilo_obj: ClCeilo | Ct25k | LufftCeilo | Cl61d | Cs135) -> int:
    is_old_chm_version = (
        hasattr(ceilo_obj, "is_old_version") and ceilo_obj.is_old_version
    )
    is_old_vaisala_model = ceilo_obj.instrument is not None and getattr(
        ceilo_obj.instrument, "model", ""
    ).lower() in ("ct25k", "cl31")
    if is_old_chm_version or is_old_vaisala_model:
        return 20
    return 5


def _initialize_ceilo(
    full_path: str | PathLike,
    site_meta: dict,
    date: datetime.date | None = None,
) -> ClCeilo | Ct25k | LufftCeilo | Cl61d | Cs135:
    if "model" in site_meta:
        if site_meta["model"] not in (
            "cl31",
            "cl51",
            "cl61d",
            "ct25k",
            "chm15k",
            "cs135",
        ):
            msg = f"Invalid ceilometer model: {site_meta['model']}"
            raise ValueError(msg)
        if site_meta["model"] in ("cl31", "cl51"):
            model = "cl31_or_cl51"
        else:
            model = site_meta["model"]
    else:
        model = _find_ceilo_model(full_path)
    if model == "cl31_or_cl51":
        return ClCeilo(full_path, site_meta, date)
    if model == "ct25k":
        return Ct25k(full_path, site_meta, date)
    if model == "cl61d":
        return Cl61d(full_path, site_meta, date)
    if model == "cs135":
        return Cs135(full_path, site_meta, date)
    return LufftCeilo(full_path, site_meta, date)


def _find_ceilo_model(full_path: str | PathLike) -> str:
    model = None
    try:
        with netCDF4.Dataset(full_path) as nc:
            title = nc.title
        for identifier in ["cl61d", "cl61-d"]:
            if (
                identifier in title.lower()
                or identifier in os.path.basename(full_path).lower()
            ):
                model = "cl61d"
        if model is None:
            model = "chm15k"
    except OSError:
        with open(full_path, "rb") as file:
            for line in islice(file, 100):
                if line.startswith(b"\x01CL"):
                    model = "cl31_or_cl51"
                elif line.startswith(b"\x01CT"):
                    model = "ct25k"
    if model is None:
        msg = "Unable to determine ceilometer model"
        raise RuntimeError(msg)
    return model


ATTRIBUTES = {
    "depolarisation": MetaData(
        long_name="Lidar volume linear depolarisation ratio",
        units="1",
        comment="SNR-screened lidar volume linear depolarisation ratio at 910.55 nm.",
        dimensions=("time", "range"),
    ),
    "depolarisation_raw": MetaData(
        long_name="Lidar volume linear depolarisation ratio",
        units="1",
        comment="SNR-screened lidar volume linear depolarisation ratio at 910.55 nm.",
        dimensions=("time", "range"),
    ),
    "calibration_factor": MetaData(
        long_name="Attenuated backscatter calibration factor",
        units="1",
        comment="Calibration factor applied.",
        dimensions=None,
    ),
    "zenith_angle": COMMON_ATTRIBUTES["zenith_angle"]._replace(dimensions=None),
}
