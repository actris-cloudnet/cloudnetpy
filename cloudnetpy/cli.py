import argparse
import base64
import concurrent.futures
import datetime
import gzip
import hashlib
import importlib
import json
import logging
import re
import shutil
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Final, Literal, cast

import requests
from cloudnet_api_client import APIClient, CloudnetAPIError
from cloudnet_api_client.containers import Instrument, ProductMetadata, RawMetadata

from cloudnetpy import concat_lib, instruments
from cloudnetpy.categorize import CategorizeInput, generate_categorize
from cloudnetpy.exceptions import PlottingError
from cloudnetpy.plotting import generate_figure

if TYPE_CHECKING:
    from collections.abc import Callable


cloudnet_api_url: Final = "https://cloudnet.fmi.fi/api/"


def run(args: argparse.Namespace, tmpdir: str, client: APIClient) -> None:
    cat_files = {}
    instrument_prefs = _parse_instrument_preferences(args.instrument)

    # Instrument based products
    if source_instruments := _get_source_instruments(
        args.products, instrument_prefs, client
    ):
        for product, possible_instruments in source_instruments.items():
            if not possible_instruments:
                continue
            meta = _fetch_raw_meta(possible_instruments, args, client)
            instrument = _select_instrument(
                meta, product, instrument_prefs.get(product)
            )
            if not instrument:
                logging.info("No instrument found for %s", product)
                continue
            meta = _filter_by_pid(meta, instrument)
            meta = _filter_by_suffix(meta, product)
            if not meta:
                logging.info("No suitable data available for %s", product)
                continue
            output_filepath = _process_instrument_product(
                product, meta, instrument, tmpdir, args, client
            )
            _plot(output_filepath, product, args)
            cat_files[product] = output_filepath

    prod_sources = _get_product_sources(args.products, client)

    # Categorize based products
    if "categorize" in args.products:
        cat_filepath = _process_categorize(cat_files, instrument_prefs, args, client)
        _plot(cat_filepath, "categorize", args)
    else:
        cat_filepath = None
    cat_products = [p for p in prod_sources if "categorize" in prod_sources[p]]
    for product in cat_products:
        if cat_filepath is None:
            cat_filepath = _fetch_product(args, "categorize", client)
        if cat_filepath is None:
            logging.info("No categorize data available for {}")
            break
        l2_filename = _process_cat_product(product, cat_filepath)
        _plot(l2_filename, product, args)

    # MWR-L1c based products
    mwrpy_products = [p for p in prod_sources if "mwr-l1c" in prod_sources[p]]
    for product in mwrpy_products:
        if "mwr-l1c" in cat_files:
            mwrpy_filepath = cat_files.get("mwr-l1c")
        else:
            mwrpy_filepath = _fetch_product(args, "mwr-l1c", client)
        if mwrpy_filepath is None:
            logging.info("No MWR-L1c data available for %s", product)
            break
        l2_filename = _process_mwrpy_product(product, mwrpy_filepath, args)
        _plot(l2_filename, product, args)


def _process_categorize(
    input_files: dict,
    instrument_prefs: dict[str, str],
    args: argparse.Namespace,
    client: APIClient,
) -> str | None:
    cat_filepath = _create_categorize_filepath(args)

    def fetch_instrument(product: str) -> tuple[str, str | None]:
        source = instrument_prefs.get(product)
        filepath = _fetch_product(args, product, client, source=source)
        if filepath is None and source:
            logging.info(
                "Preferred instrument '%s' not found for %s, using available",
                source,
                product,
            )
            filepath = _fetch_product(args, product, client)
        return product, filepath

    def fetch_mwr() -> tuple[str, str | None]:
        try:
            return "mwr", _fetch_mwr(args, client)
        except (CloudnetAPIError, KeyError, ValueError):
            logging.info("LWP data not available, continuing without it")
            return "mwr", None

    def fetch_model() -> tuple[str, str | None]:
        return "model", _fetch_model(args, client)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(fetch_model)]
        for product in ("radar", "lidar", "disdrometer"):
            if product not in input_files:
                futures.append(executor.submit(fetch_instrument, product))

        for future in concurrent.futures.as_completed(futures):
            product, filepath = future.result()
            if filepath:
                input_files[product] = filepath

    # MWR runs after other downloads because its fallback chain may
    # try to download the same radar file fetched above.
    mwr_key, mwr_path = fetch_mwr()
    if mwr_path:
        input_files[mwr_key] = mwr_path

    if "model" not in input_files:
        logging.info("No model data available for this date.")
        return None

    try:
        logging.info("Processing categorize...")
        generate_categorize(
            cast("CategorizeInput", input_files),
            cat_filepath,
            options=args.options,
        )
        logging.info("Processed categorize to %s", cat_filepath)
    except NameError:
        logging.info("No data available for this date.")
        return None
    return cat_filepath


def _fetch_mwr(args: argparse.Namespace, client: APIClient) -> str | None:
    mwr_sources = [
        ("mwr-single", None),
        ("mwr", None),
        ("radar", "rpg-fmcw-35"),
        ("radar", "rpg-fmcw-94"),
    ]
    for product, source in mwr_sources:
        mwr = _fetch_product(args, product, client, source=source)
        if mwr:
            return mwr
    return None


def _process_instrument_product(
    product: str,
    meta: list[RawMetadata],
    instrument: Instrument,
    tmpdir: str,
    args: argparse.Namespace,
    client: APIClient,
) -> str | None:
    output_filepath = _create_instrument_filepath(instrument, args)
    site_obj = meta[0].site
    site_meta: dict = {
        "name": site_obj.human_readable_name,
        "latitude": site_obj.latitude,
        "longitude": site_obj.longitude,
        "altitude": site_obj.altitude,
    }
    input_files: list[Path] | Path
    input_files = _fetch_raw(meta, args, client)
    input_files = [_unzip_gz_file(f) for f in input_files]
    if args.dl:
        return None
    input_folder = input_files[0].parent
    calibration = _get_calibration(instrument, args, client)
    fun: Callable
    match (product, instrument.instrument_id):
        case ("radar", _id) if "mira" in _id:
            site_meta["range_correction_factor"] = calibration.get(
                "range_correction_factor"
            )
            fun = instruments.mira2nc
        case ("radar", _id) if "rpg" in _id:
            fun = instruments.rpg2nc
            input_files = input_folder
        case ("radar", _id) if "basta" in _id:
            fun = instruments.basta2nc
            _check_input(input_files)
            input_files = input_files[0]
        case ("radar", _id) if "copernicus" in _id:
            fun = instruments.copernicus2nc
        case ("radar", _id) if "galileo" in _id:
            fun = instruments.galileo2nc
        case ("disdrometer", _id) if "parsivel" in _id:
            fun = instruments.parsivel2nc
        case ("disdrometer", _id) if "thies" in _id:
            fun = instruments.thies2nc
            input_files = _concatenate_(input_files, tmpdir)
        case ("lidar", _id) if "pollyxt" in _id:
            site_meta["snr_limit"] = calibration.get("snr_limit", 25)
            fun = instruments.pollyxt2nc
        case ("lidar", _id) if _id == "cl61d":
            fun = instruments.ceilo2nc
            variables = ["x_pol", "p_pol", "beta_att", "time", "tilt_angle"]
            concat_file = Path(tmpdir) / "tmp.nc"
            concat_lib.bundle_netcdf_files(
                input_files,
                datetime.date.fromisoformat(args.date),
                concat_file,
                variables=variables,
            )
            input_files = concat_file
            site_meta["model"] = instrument.instrument_id
        case ("lidar", _id):
            fun = instruments.ceilo2nc
            input_files = _concatenate_(input_files, tmpdir)
            site_meta["model"] = instrument.instrument_id
            if factor := calibration.get("calibration_factor"):
                site_meta["calibration_factor"] = factor
        case ("mwr", _id):
            fun = instruments.hatpro2nc
            input_files = input_folder
        case ("mwr-l1c", _id):
            fun = instruments.hatpro2l1c
            site_meta = {**site_meta, **calibration}
            coefficients, links = _fetch_coefficient_files(calibration, tmpdir)
            site_meta["coefficientFiles"] = coefficients
            site_meta["coefficientLinks"] = links
            input_files = input_folder
        case ("mrr", _id):
            fun = instruments.mrr2nc
        case ("weather-station", _id):
            fun = instruments.ws2nc
    logging.info("Processing %s...", product)
    fun(input_files, output_filepath, site_meta, date=args.date)
    logging.info("Processed %s: %s", product, output_filepath)
    return output_filepath


def _concatenate_(input_files: list[Path], tmpdir: str) -> Path:
    if len(input_files) > 1:
        concat_file = Path(tmpdir) / "tmp.nc"
        try:
            concat_lib.concatenate_files(input_files, concat_file)
        except OSError:
            concat_lib.concatenate_text_files(input_files, concat_file)
        return concat_file
    return input_files[0]


def _fetch_coefficient_files(
    calibration: dict, tmpdir: str
) -> tuple[list[str], list[str]]:
    msg = "No calibration coefficients found"
    if not (coeffs := calibration.get("retrieval_coefficients")):
        raise ValueError(msg)
    if not (links := coeffs[0].get("links")):
        raise ValueError(msg)
    coefficient_paths = []
    for filename in links:
        res = requests.get(filename, timeout=60)
        res.raise_for_status()
        filepath = Path(tmpdir) / Path(filename).name
        filepath.write_bytes(res.content)
        coefficient_paths.append(str(filepath))
    return coefficient_paths, links


def _get_calibration(
    instrument: Instrument, args: argparse.Namespace, client: APIClient
) -> dict:
    try:
        calibration = client.calibration(
            date=args.date,
            instrument_pid=instrument.pid,
        )
        return calibration.get("data", {})
    except CloudnetAPIError:
        return {}


def _create_instrument_filepath(
    instrument: Instrument, args: argparse.Namespace
) -> str:
    folder = _create_output_folder("instrument", args)
    pid = _shorten_pid(instrument.pid)
    filename = (
        f"{args.date.replace('-', '')}_{args.site}_{instrument.instrument_id}_{pid}.nc"
    )
    return str(folder / filename)


def _create_categorize_filepath(args: argparse.Namespace) -> str:
    folder = _create_output_folder("geophysical", args)
    filename = f"{args.date.replace('-', '')}_{args.site}_categorize.nc"
    return str(folder / filename)


def _create_input_folder(end_point: str, args: argparse.Namespace) -> Path:
    folder = args.input / args.site / args.date / end_point
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _create_output_folder(end_point: str, args: argparse.Namespace) -> Path:
    folder = args.output / args.site / args.date / end_point
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _fetch_raw_meta(
    instruments: list[str], args: argparse.Namespace, client: APIClient
) -> list[RawMetadata]:
    return client.raw_files(
        site_id=args.site,
        date=args.date,
        instrument_id=instruments,
        status=["uploaded", "processed"],
    )


def _filter_by_pid(
    meta: list[RawMetadata], instrument: Instrument
) -> list[RawMetadata]:
    return [m for m in meta if m.instrument.pid == instrument.pid]


def _filter_by_suffix(meta: list[RawMetadata], product: str) -> list[RawMetadata]:
    if product == "radar":
        meta = [m for m in meta if not m.filename.lower().endswith(".lv0")]
    elif product == "mwr":
        meta = [m for m in meta if re.search(r"\.(lwp|iwv)", m.filename, re.IGNORECASE)]
    elif product == "mwr-l1c":
        meta = [m for m in meta if not m.filename.lower().endswith(".nc")]
    return meta


def _get_source_instruments(
    products: list[str], instrument_prefs: dict[str, str], client: APIClient
) -> dict[str, list[str]]:
    source_instruments = {}
    for product in products:
        prod, model = _parse_instrument(product)
        if model is None:
            pref = instrument_prefs.get(prod)
            if pref is not None and not _is_pid(pref):
                model = pref
        all_possible = client.product(prod).source_instrument_ids
        if all_possible and (match := [i for i in all_possible if i == model]):
            source_instruments[prod] = match
        else:
            source_instruments[prod] = list(all_possible)
    return source_instruments


def _is_pid(value: str) -> bool:
    return value.startswith("https://hdl.handle.net/")


def _instrument_matches(instrument: Instrument, value: str) -> bool:
    if _is_pid(value):
        return instrument.pid == value
    return instrument.instrument_id == value


def _get_product_sources(
    products: list[str], client: APIClient
) -> dict[str, list[str]]:
    source_products = {}
    for product in products:
        prod, _ = _parse_instrument(product)
        product_obj = client.product(prod)
        if product_obj.source_product_ids:
            source_products[prod] = list(product_obj.source_product_ids)
    return source_products


def _parse_instrument_preferences(args: list[str] | None) -> dict[str, str]:
    """Parses instrument preferences like 'radar:mira-35' into a dict."""
    if not args:
        return {}
    prefs = {}
    for arg in args:
        if ":" not in arg:
            msg = (
                f"Invalid instrument format '{arg}', "
                "expected 'product:instrument_id' or 'product:pid'"
            )
            raise argparse.ArgumentTypeError(msg)
        product, value = arg.split(":", 1)
        prefs[product] = value
    return prefs


def _parse_instrument(s: str) -> tuple[str, str | None]:
    if "[" in s and s.endswith("]"):
        name = s[: s.index("[")]
        value = s[s.index("[") + 1 : -1]
    else:
        name = s
        value = None
    return name, value


def _select_instrument(
    meta: list[RawMetadata], product: str, pref: str | None = None
) -> Instrument | None:
    instruments = _get_unique_instruments(meta)
    if len(instruments) == 0:
        logging.info("No instruments found")
        return None
    if pref:
        matches = [i for i in instruments if _instrument_matches(i, pref)]
        if matches:
            logging.info("Selected instrument by preference: %s", matches[0].name)
            return matches[0]
        logging.info("Preferred instrument '%s' not found, falling back", pref)
    if len(instruments) > 1:
        logging.info("Multiple instruments found for %s", product)
        logging.info("Please specify which one to use")
        for i, instrument in enumerate(instruments):
            logging.info("%d: %s (%s)", i + 1, instrument.name, instrument.pid)
        try:
            ind = int(input("Select: ")) - 1
        except (ValueError, EOFError, KeyboardInterrupt):
            logging.info("Selection cancelled")
            return None
        if not 0 <= ind < len(instruments):
            logging.info("Invalid selection")
            return None
        selected_instrument = instruments[ind]
    else:
        selected_instrument = instruments[0]
        logging.info("Single instrument found: %s", selected_instrument.name)
    return selected_instrument


def _get_unique_instruments(meta: list[RawMetadata]) -> list[Instrument]:
    unique_instruments = list({m.instrument for m in meta})
    return sorted(unique_instruments, key=lambda x: x.name)


def _fetch_product(
    args: argparse.Namespace, product: str, client: APIClient, source: str | None = None
) -> str | None:
    meta = client.files(product_id=product, date=args.date, site_id=args.site)
    if source:
        meta = [
            m
            for m in meta
            if m.instrument is not None and _instrument_matches(m.instrument, source)
        ]
    if not meta:
        return None
    if len(meta) > 1:
        logging.info(
            "Multiple files for %s ... taking the first but some logic needed", product
        )
        meta = [meta[0]]
    suffix = "geophysical" if "geophysical" in meta[0].product.type else "instrument"
    folder = _create_output_folder(suffix, args)
    return _download_product_file(meta, folder, client, force=args.force_download)


def _fetch_model(args: argparse.Namespace, client: APIClient) -> str | None:
    files = client.files(
        product_id="model", model_id=args.model, date=args.date, site_id=args.site
    )
    if not files:
        logging.info("No model data available for this date")
        return None
    folder = _create_output_folder("instrument", args)
    return _download_product_file(files, folder, client, force=args.force_download)


def _fetch_raw(
    metadata: list[RawMetadata], args: argparse.Namespace, client: APIClient
) -> list[Path]:
    pid = _shorten_pid(metadata[0].instrument.pid)
    instrument = f"{metadata[0].instrument.instrument_id}_{pid}"
    folder = _create_input_folder(instrument, args)
    if args.force_download:
        for meta in metadata:
            filepath = folder / meta.download_url.split("/")[-1]
            if filepath.exists():
                logging.info("Removing existing file: %s", filepath)
                filepath.unlink()
    return client.download(metadata, output_directory=folder)


def _download_product_file(
    meta: list[ProductMetadata],
    folder: Path,
    client: APIClient,
    *,
    force: bool = False,
) -> str:
    if len(meta) > 1:
        msg = "Multiple product files found"
        raise ValueError(msg)
    filepath = folder / meta[0].filename
    if filepath.exists() and not force:
        logging.info("Existing file found: %s", filepath)
        return str(filepath)
    if filepath.exists():
        logging.info("Removing existing file: %s", filepath)
        filepath.unlink()
    logging.info("Downloading file: %s", filepath)
    return str(client.download(meta, output_directory=folder)[0])


def _shorten_pid(pid: str) -> str:
    return pid.split(".")[-1][:8]


def _check_input(files: list) -> None:
    if len(files) > 1:
        msg = "Multiple input files found"
        raise ValueError(msg)


def _plot(
    filepath: PathLike | str | None, product: str, args: argparse.Namespace
) -> None:
    if filepath is None or (not args.plot and not args.show):
        return
    if args.variables is not None:
        variables = args.variables.split(",")
    else:
        res = requests.get(f"{cloudnet_api_url}products/variables", timeout=60)
        res.raise_for_status()
        variables = next(var["variables"] for var in res.json() if var["id"] == product)
        variables = [var["id"].split("-")[-1] for var in variables]
    image_name = str(filepath).replace(".nc", ".png") if args.plot else None
    try:
        generate_figure(
            filepath,
            variables,
            show=args.show,
            output_filename=image_name,
        )
    except PlottingError as e:
        logging.info("Failed to plot %s: %s", product, e)
    if args.plot:
        logging.info("Plotted %s: %s", product, image_name)


def _process_cat_product(product: str, categorize_file: str) -> str:
    output_file = categorize_file.replace("categorize", product)
    module = importlib.import_module("cloudnetpy.products")
    getattr(module, f"generate_{product}")(categorize_file, output_file)
    logging.info("Processed %s: %s", product, output_file)
    return output_file


def _process_mwrpy_product(
    product: str, mwr_l1c_file: str, args: argparse.Namespace
) -> str:
    filename = f"{args.date}_{args.site}_{product}.nc"
    output_file = _create_output_folder("geophysical", args) / filename
    module = importlib.import_module("cloudnetpy.products")
    getattr(module, f"generate_{product.replace('-', '_')}")(mwr_l1c_file, output_file)
    logging.info("Processed %s: %s", product, output_file)
    return str(output_file)


def _parse_products(product_argument: str, client: APIClient) -> list[str]:
    products = product_argument.split(",")
    valid_options = [p.id for p in client.products()]
    valid_products = []
    for product in products:
        prod, _ = _parse_instrument(product)
        if prod in valid_options:
            valid_products.append(product)
    return valid_products


def _parse_model(model_id: str, client: APIClient) -> str:
    models = client.models()
    found = False
    for model in models:
        if model_id == model.source_model_id and model.id != model.source_model_id:
            msg = f"Use {model.id} or similar instead of {model_id}"
            raise argparse.ArgumentTypeError(msg)
        if model_id == model.id:
            found = True
    if not found:
        msg = f"Invalid model: {model_id}"
        raise argparse.ArgumentTypeError(msg)
    return model_id


def main() -> None:
    client = APIClient()
    parser = argparse.ArgumentParser(
        description="Command line interface for running CloudnetPy."
    )
    parser.add_argument(
        "-s",
        "--site",
        type=str,
        help="Site",
        required=True,
        choices=[site.id for site in client.sites()],
        metavar="SITE",
    )
    parser.add_argument(
        "-d",
        "--date",
        type=str,
        help="Date in YYYY-MM-DD (default: today)",
        default=datetime.datetime.now(tz=datetime.timezone.utc).date().isoformat(),
    )
    parser.add_argument(
        "-p",
        "--products",
        type=lambda arg: _parse_products(arg, client),
        help=(
            "Products to process, e.g. 'radar' or 'classification'. If the site "
            "has many instruments, you can specify the instrument in brackets, "
            "e.g. radar[mira-35]."
        ),
        required=True,
    )
    parser.add_argument(
        "-m",
        "--model",
        type=lambda arg: _parse_model(arg, client),
        help="Model to use in categorize.",
    )
    parser.add_argument(
        "-i",
        "--instrument",
        type=str,
        action="append",
        help=(
            "Preferred instrument for a product, e.g. 'radar:mira-35', "
            "'lidar:cl61d', or 'radar:https://hdl.handle.net/<pid>'. The "
            "value is either an instrument_id or an instrument PID. Applies "
            "both to instrument processing and categorize input. Can be "
            "specified multiple times."
        ),
        default=None,
    )
    parser.add_argument("--input", type=Path, help="Input path", default="input/")
    parser.add_argument("--output", type=Path, help="Output path", default="output/")
    parser.add_argument(
        "--plot",
        help="Plot the processed data",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--show",
        help="Show plotted image",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--dl",
        help="Download raw data only",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--force-download",
        help="Force re-download of input files even if they exist locally",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "-v",
        "--variables",
        type=str,
        help="Variables to plot (comma-separated), e.g. 'target_classification'",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--options",
        type=json.loads,
        help=(
            "Options for categorize processing as a JSON string, "
            "e.g. '{\"temperature_offset\": 3}'"
        ),
        default=None,
    )
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = [handler]

    with TemporaryDirectory() as tmpdir:
        run(args, tmpdir, client)


def md5sum(filename: str | PathLike, *, is_base64: bool = False) -> str:
    """Calculates hash of file using md5."""
    return _calc_hash_sum(filename, "md5", is_base64=is_base64)


def _calc_hash_sum(
    filename: str | PathLike, method: Literal["sha256", "md5"], *, is_base64: bool
) -> str:
    hash_sum = getattr(hashlib, method)()
    with open(filename, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            hash_sum.update(byte_block)
    if is_base64:
        return base64.encodebytes(hash_sum.digest()).decode("utf-8").strip()
    return hash_sum.hexdigest()


def _unzip_gz_file(path_in: Path) -> Path:
    if path_in.suffix != ".gz":
        return path_in
    path_out = path_in.with_suffix("")
    logging.debug("Decompressing %s to %s", path_in, path_out)
    with gzip.open(path_in, "rb") as file_in, open(path_out, "wb") as file_out:
        shutil.copyfileobj(file_in, file_out)
    return path_out


if __name__ == "__main__":
    main()
