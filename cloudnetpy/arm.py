"""Module for fetching and converting ARM data."""

import concurrent.futures
import datetime
import functools
import logging
import os
from collections.abc import Callable, Iterable
from os import PathLike
from pathlib import Path
from typing import Final, NamedTuple

import netCDF4
import requests

from cloudnetpy.exceptions import CloudnetException
from cloudnetpy.instruments import (
    arm_ceilo,
    armceilo2nc,
    armld2nc,
    kazr2nc,
    mmcr2nc,
    mwr3c2nc,
    mwrlos2nc,
)

ARM_LIVE_URL: Final = "https://adc.arm.gov/armlive/livedata"
ARM_USER_ENV: Final = "ARM_USER"
ARM_TOKEN_ENV: Final = "ARM_TOKEN"  # noqa: S105
HTTP_RANGE_NOT_SATISFIABLE: Final = 416
MAX_PARALLEL_DOWNLOADS: Final = 4
CEILOMETER_MODELS: Final = ("ct25k", "cl31", "cl51")

# Cloudnet site id -> (ARM site code, facility)
ARM_SITES: Final = {
    "arm-andoya": ("anx", "M1"),
    "arm-ascension": ("asi", "M1"),
    "arm-aware": ("awr", "M1"),
    "arm-cape-cod": ("pvc", "M1"),
    "arm-darwin": ("twp", "C3"),
    "arm-ganges": ("pgh", "M1"),
    "arm-graciosa": ("ena", "C1"),
    "arm-macquarie": ("maq", "M1"),
    "arm-maldives": ("gan", "M1"),
    "arm-manacapuru": ("mao", "M1"),
    "arm-murgtal": ("fkb", "M1"),
    "arm-niamey": ("nim", "M1"),
    "arm-nsa": ("nsa", "C1"),
    "arm-oliktok": ("oli", "M1"),
    "arm-sgp": ("sgp", "C1"),
    "arm-yacanto": ("cor", "M1"),
}


class ArmInstrument(NamedTuple):
    product: str
    instrument_id: str
    datastream: str  # without site code and facility
    reader: Callable
    multi_file: bool  # reader accepts a list of files


# In order of preference within each product; the first with files is used.
ARM_INSTRUMENTS: Final = (
    # KAZR2 (approx. 2019 onwards, hourly files)
    ArmInstrument("radar", "kazr", "kazrcfrge.a1", kazr2nc, multi_file=True),
    # KAZR2 at ENA (Dec 2019 onwards, hourly files)
    ArmInstrument("radar", "kazr", "kazr2cfrge.a1", kazr2nc, multi_file=True),
    # KAZR corrected moments (approx. 2011-2019)
    ArmInstrument("radar", "kazr", "kazrcorge.c1", kazr2nc, multi_file=True),
    # KAZR moments (approx. 2011-2019, e.g. mobile facilities)
    ArmInstrument("radar", "kazr", "kazrge.a1", kazr2nc, multi_file=True),
    # MMCR (before 2011)
    ArmInstrument("radar", "mmcr", "mmcrmom.b1", mmcr2nc, multi_file=False),
    ArmInstrument("lidar", "cl31", "ceil.b1", armceilo2nc, multi_file=False),
    ArmInstrument("mwr", "wvr-1100", "mwrlos.b1", mwrlos2nc, multi_file=True),
    # 3-channel MWR (e.g. ENA)
    ArmInstrument("mwr", "mwr-3c", "mwr3c.b1", mwr3c2nc, multi_file=True),
    ArmInstrument("disdrometer", "parsivel", "ld.b1", armld2nc, multi_file=False),
)


class ArmDataError(CloudnetException):
    """Raised when ARM data can not be fetched."""


class ArmFileNotFoundError(ArmDataError):
    """Raised when a file listed in the ARM catalog is not served."""


def is_arm_site(site_id: str) -> bool:
    return site_id in ARM_SITES


def get_datastream(site_id: str, instrument: ArmInstrument) -> str:
    """Returns full ARM datastream name, e.g. 'sgpmmcrmomC1.b1'."""
    try:
        site_code, facility = ARM_SITES[site_id]
    except KeyError as err:
        msg = f"Unknown ARM site: {site_id}"
        raise ArmDataError(msg) from err
    name, level = instrument.datastream.split(".")
    return f"{site_code}{name}{facility}.{level}"


def get_credentials() -> str:
    """Returns ARM Live 'user:token' string from environment variables."""
    user = os.environ.get(ARM_USER_ENV)
    token = os.environ.get(ARM_TOKEN_ENV)
    if not user or not token:
        msg = (
            f"ARM Live credentials missing: set {ARM_USER_ENV} and {ARM_TOKEN_ENV} "
            "environment variables (https://adc.arm.gov/armlive/)"
        )
        raise ArmDataError(msg)
    return f"{user}:{token}"


def query_files(
    datastream: str, date: datetime.date, credentials: str | None = None
) -> list[str]:
    """Lists ARM files of a datastream for one date."""
    credentials = credentials or get_credentials()
    params = {
        "user": credentials,
        "ds": datastream,
        "start": date.isoformat(),
        "end": date.isoformat(),
        "wt": "json",
    }
    res = requests.get(f"{ARM_LIVE_URL}/query", params=params, timeout=60)
    if not res.ok:
        msg = f"ARM Live query of {datastream} failed: HTTP {res.status_code}"
        raise ArmDataError(msg)
    try:
        data = res.json()
    except requests.JSONDecodeError as err:
        msg = f"Invalid response from ARM Live: {res.text[:200]}"
        raise ArmDataError(msg) from err
    if data.get("status") != "success":
        msg = f"ARM Live query failed: {data}"
        raise ArmDataError(msg)
    return sorted(data.get("files", []))


def download_file(
    filename: str,
    output_dir: str | PathLike,
    credentials: str | None = None,
    *,
    force: bool = False,
    max_attempts: int = 5,
) -> Path:
    """Downloads one ARM file, resuming interrupted transfers.

    ARM Live may drop large transfers. The file is downloaded into a temporary
    `.part` file which is resumed with HTTP range requests until complete.
    """
    credentials = credentials or get_credentials()
    filepath = Path(output_dir) / filename
    part = filepath.with_name(filename + ".part")
    params = {"user": credentials, "file": filename}
    remote_size = None
    if filepath.exists() and not force:
        remote_size = _get_remote_size(params)
        if remote_size is None or filepath.stat().st_size == remote_size:
            logging.info("Existing file found: %s", filepath)
            return filepath
        # Truncated file from an interrupted download: resume it
        logging.warning("Existing file %s is incomplete", filepath)
        filepath.replace(part)
    if force and part.exists():
        part.unlink()
    if part.exists():
        remote_size = remote_size or _get_remote_size(params)
        if remote_size is not None and part.stat().st_size == remote_size:
            part.replace(filepath)
            return filepath
        if remote_size is not None and part.stat().st_size > remote_size:
            part.unlink()
    logging.info("Downloading file: %s", filepath)
    total = None
    for _ in range(max_attempts):
        offset = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset > 0 else {}
        if offset > 0:
            logging.info("Resuming download of %s from %s bytes", filename, offset)
        try:
            with requests.get(
                f"{ARM_LIVE_URL}/saveData",
                params=params,
                headers=headers,
                timeout=(60, 600),
                stream=True,
            ) as res:
                res.raise_for_status()
                if offset > 0 and res.status_code != requests.codes.partial_content:
                    offset = 0  # Server ignored the range: start over
                total = _get_total_size(res, offset)
                with open(part, "ab" if offset > 0 else "wb") as f:
                    f.writelines(res.iter_content(chunk_size=1 << 20))
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
        ) as err:
            logging.warning("Download interrupted: %s", err)
            continue
        except requests.HTTPError as err:
            status = err.response.status_code
            if status == requests.codes.not_found:
                msg = f"File {filename} not found on ARM Live"
                raise ArmFileNotFoundError(msg) from None
            if status != HTTP_RANGE_NOT_SATISFIABLE:
                msg = f"Download of {filename} failed: HTTP {status}"
                raise ArmDataError(msg) from None
            logging.warning("Can not resume download of %s, restarting", filename)
            part.unlink()
            continue
        size = part.stat().st_size
        if total is None or size == total:
            part.replace(filepath)
            return filepath
        if size > total:
            part.unlink()
    msg = f"Incomplete download of {filename}"
    raise ArmDataError(msg)


def _get_remote_size(params: dict) -> int | None:
    """Returns the size of a file on ARM Live, or None if unknown."""
    try:
        res = requests.get(
            f"{ARM_LIVE_URL}/saveData",
            params=params,
            headers={"Range": "bytes=0-0"},
            timeout=60,
            stream=True,
        )
        res.close()
    except requests.RequestException:
        return None
    if not res.ok:
        return None
    return _get_total_size(res, 0)


def _get_total_size(res: requests.Response, offset: int) -> int | None:
    """Returns total file size from response headers, if available."""
    content_range = res.headers.get("Content-Range")
    if content_range and "/" in content_range:
        return int(content_range.split("/")[-1])
    content_length = res.headers.get("Content-Length")
    if content_length is not None:
        return offset + int(content_length)
    return None


def fetch_files(
    site_id: str,
    date: datetime.date,
    output_dir: str | PathLike,
    products: Iterable[str] = ("radar", "lidar", "mwr", "disdrometer"),
    *,
    force: bool = False,
) -> dict[str, list[Path]]:
    """Downloads ARM raw files of one date.

    Args:
        site_id: Cloudnet site id, e.g. 'arm-sgp'.
        date: Date to fetch.
        output_dir: Files are saved into `output_dir/<datastream>/`.
        products: Cloudnet products to fetch files for.
        force: Re-download existing files.

    Returns:
        Downloaded files per product, e.g. {'radar': [Path(...)]}.

    """
    credentials = get_credentials()
    files: dict[str, list[Path]] = {}
    for instrument in ARM_INSTRUMENTS:
        if instrument.product not in products or instrument.product in files:
            continue
        datastream = get_datastream(site_id, instrument)
        filenames = query_files(datastream, date, credentials)
        if not filenames:
            logging.info("No %s data found for %s", datastream, date)
            continue
        folder = Path(output_dir) / datastream
        folder.mkdir(parents=True, exist_ok=True)
        with concurrent.futures.ThreadPoolExecutor(MAX_PARALLEL_DOWNLOADS) as executor:
            download = functools.partial(
                download_file, output_dir=folder, credentials=credentials, force=force
            )
            try:
                files[instrument.product] = list(executor.map(download, filenames))
            except ArmFileNotFoundError:
                # Listed in the ARM catalog but not served: try the next instrument
                logging.warning("%s data not available for download", datastream)
    return files


def find_l1b_files(
    site_id: str, date: datetime.date, output_dir: str | PathLike
) -> dict[str, str]:
    """Finds existing Level 1b files created by `convert_to_l1b`."""
    files = {}
    for instrument in ARM_INSTRUMENTS:
        if instrument.product in files:
            continue
        for instrument_id in _instrument_ids(instrument):
            filepath = Path(output_dir) / _l1b_filename(site_id, date, instrument_id)
            if filepath.exists():
                files[instrument.product] = str(filepath)
                break
    return files


def _instrument_ids(instrument: ArmInstrument) -> tuple[str, ...]:
    """Possible instrument ids of a datastream (ceilometer model varies)."""
    if instrument.reader is armceilo2nc:
        return CEILOMETER_MODELS
    return (instrument.instrument_id,)


def _instrument_id(instrument: ArmInstrument, raw_file: Path) -> str:
    """Instrument id of a raw file, read from the file for ceilometers."""
    if instrument.reader is armceilo2nc:
        with netCDF4.Dataset(raw_file) as nc:
            return arm_ceilo.read_model(nc).lower()
    return instrument.instrument_id


def _l1b_filename(site_id: str, date: datetime.date, instrument_id: str) -> str:
    return f"{date.strftime('%Y%m%d')}_{site_id}_{instrument_id}.nc"


def convert_to_l1b(
    site_id: str,
    date: datetime.date,
    files: dict[str, list[Path]],
    output_dir: str | PathLike,
    site_meta: dict,
    calibration: dict[str, dict] | None = None,
) -> dict[str, str]:
    """Converts ARM raw files into Cloudnet Level 1b files.

    Args:
        site_id: Cloudnet site id, e.g. 'arm-sgp'.
        date: Date of the files.
        files: Raw files per product, as returned by `fetch_files`.
        output_dir: Folder for the output files.
        site_meta: Site metadata with at least `name`.
        calibration: Optional reader options per product, e.g.
            {'lidar': {'calibration_factor': 1.2}, 'radar': {'mode': 'GE'}}.

    Returns:
        Level 1b file per product, e.g. {'radar': 'path/to/radar.nc'}.

    """
    calibration = calibration or {}
    l1b_files = {}
    for product, raw_files in files.items():
        if not raw_files:
            continue
        instrument = _find_instrument(site_id, product, raw_files[0])
        instrument_id = _instrument_id(instrument, raw_files[0])
        output_file = Path(output_dir) / _l1b_filename(site_id, date, instrument_id)
        meta = {**site_meta, **calibration.get(product, {})}
        input_files: Path | list[Path] = raw_files[0]
        if len(raw_files) > 1:
            if instrument.multi_file:
                input_files = raw_files
            else:
                input_files = max(raw_files, key=lambda f: f.stat().st_size)
                logging.warning(
                    "Found %s %s files, using only %s",
                    len(raw_files),
                    product,
                    input_files.name,
                )
        try:
            instrument.reader(input_files, output_file, meta, date=date)
        except (CloudnetException, KeyError, ValueError, RuntimeError, OSError) as err:
            logging.warning("Failed to process %s: %s", product, err)
            continue
        logging.info("Processed %s: %s", product, output_file)
        l1b_files[product] = str(output_file)
    return l1b_files


def _find_instrument(site_id: str, product: str, filename: Path) -> ArmInstrument:
    """Finds the instrument definition matching an ARM filename."""
    for instrument in ARM_INSTRUMENTS:
        if instrument.product != product:
            continue
        if filename.name.startswith(get_datastream(site_id, instrument)):
            return instrument
    msg = f"Unknown ARM {product} file: {filename.name}"
    raise ArmDataError(msg)
