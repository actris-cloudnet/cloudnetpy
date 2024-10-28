import argparse
import gzip
import importlib
import logging
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Final

import requests

from cloudnetpy import concat_lib, instruments
from cloudnetpy.categorize import generate_categorize
from cloudnetpy.exceptions import PlottingError
from cloudnetpy.plotting import generate_figure
from cloudnetpy.utils import md5sum

if TYPE_CHECKING:
    from collections.abc import Callable


cloudnet_api_url: Final = "https://cloudnet.fmi.fi/api/"


@dataclass
class Instrument:
    id: str
    pid: str
    name: str


def run(args: argparse.Namespace, tmpdir: str):
    cat_files = {}

    # Instrument based products
    if source_instruments := _get_source_instruments(args.products):
        for product, possible_instruments in source_instruments.items():
            meta = _fetch_raw_meta(possible_instruments, args)
            instrument = _select_instrument(meta, product)
            if not instrument:
                logging.info("No instrument found for %s", product)
                continue
            meta = _filter_by_instrument(meta, instrument)
            meta = _filter_by_suffix(meta, product)
            if not meta:
                logging.info("No suitable data available for %s", product)
                continue
            output_filepath = _process_instrument_product(
                product, meta, instrument, tmpdir, args
            )
            _plot(output_filepath, product, args)
            cat_files[product] = output_filepath

    prod_sources = _get_product_sources(args.products)

    # Categorize based products
    if "categorize" in args.products:
        cat_filepath = _process_categorize(cat_files, args)
        _plot(cat_filepath, "categorize", args)
    else:
        cat_filepath = None
    cat_products = [p for p in prod_sources if "categorize" in prod_sources[p]]
    for product in cat_products:
        if cat_filepath is None:
            cat_filepath = _fetch_product(args, "categorize")
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
            mwrpy_filepath = _fetch_product(args, "mwr-l1c")
        if mwrpy_filepath is None:
            logging.info("No MWR-L1c data available for %s", product)
            break
        l2_filename = _process_mwrpy_product(product, mwrpy_filepath, args)
        _plot(l2_filename, product, args)


def _process_categorize(input_files: dict, args: argparse.Namespace) -> str | None:
    cat_filepath = _create_categorize_filepath(args)

    input_files["model"] = _fetch_model(args)
    if input_files["model"] is None:
        logging.info("No model data available for this date.")
        return None

    for product in ("radar", "lidar", "disdrometer"):
        if product not in input_files and (filepath := _fetch_product(args, product)):
            input_files[product] = filepath

    if mwr := _fetch_mwr(args):
        input_files["mwr"] = mwr

    try:
        logging.info("Processing categorize...")
        generate_categorize(input_files, cat_filepath)
        logging.info("Processed categorize to %s", cat_filepath)
    except NameError:
        logging.info("No data available for this date.")
        return None
    return cat_filepath


def _fetch_mwr(args: argparse.Namespace) -> str | None:
    mwr_sources = [
        ("mwr-single", None),
        ("mwr", None),
        ("radar", "rpg-fmcw-35"),
        ("radar", "rpg-fmcw-94"),
    ]
    for product, source in mwr_sources:
        mwr = _fetch_product(args, product, source=source)
        if mwr:
            return mwr
    return None


def _process_instrument_product(
    product: str,
    meta: list[dict],
    instrument: Instrument,
    tmpdir: str,
    args: argparse.Namespace,
) -> str | None:
    output_filepath = _create_instrument_filepath(instrument, args)
    site_meta = _read_site_meta(meta)
    input_files: list[str] | str
    input_files = _fetch_raw(meta, args)
    if args.dl:
        return None
    input_folder = str(Path(input_files[0]).parent)
    calibration = _get_calibration(instrument, args)
    fun: Callable
    match (product, instrument.id):
        case ("radar", _id) if "mira" in _id:
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
            fun = instruments.pollyxt2nc
        case ("lidar", _id) if _id == "cl61d":
            fun = instruments.ceilo2nc
            variables = ["x_pol", "p_pol", "beta_att", "time", "tilt_angle"]
            concat_file = str(Path(tmpdir) / "tmp.nc")
            concat_lib.bundle_netcdf_files(
                input_files,
                args.date,
                concat_file,
                variables=variables,
            )
            input_files = concat_file
            site_meta["model"] = instrument.id
        case ("lidar", _id):
            fun = instruments.ceilo2nc
            input_files = _concatenate_(input_files, tmpdir)
            site_meta["model"] = instrument.id
            if factor := calibration.get("calibration_factor"):
                site_meta["calibration_factor"] = factor
        case ("mwr", _id):
            fun = instruments.hatpro2nc
            input_files = input_folder
        case ("mwr-l1c", _id):
            fun = instruments.hatpro2l1c
            coefficients = _fetch_coefficient_files(calibration, tmpdir)
            site_meta = {**site_meta, **calibration}
            site_meta["coefficientFiles"] = coefficients
            input_files = input_folder
        case ("mrr", _id):
            fun = instruments.mrr2nc
        case ("weather-station", _id):
            fun = instruments.ws2nc
    logging.info("Processing %s...", product)
    fun(input_files, output_filepath, site_meta, date=args.date)
    logging.info("Processed %s: %s", product, output_filepath)
    return output_filepath


def _concatenate_(input_files: list[str], tmpdir: str) -> str:
    if len(input_files) > 1:
        concat_file = str(Path(tmpdir) / "tmp.nc")
        try:
            concat_lib.concatenate_files(input_files, concat_file)
        except OSError:
            concat_lib.concatenate_text_files(input_files, concat_file)
        return concat_file
    return input_files[0]


def _fetch_coefficient_files(calibration: dict, tmpdir: str) -> list:
    if not (links := calibration.get("coefficientLinks")):
        msg = "No calibration coefficients found"
        raise ValueError(msg)
    coefficient_paths = []
    for filename in links:
        res = requests.get(filename, timeout=60)
        res.raise_for_status()
        filepath = Path(tmpdir) / Path(filename).name
        filepath.write_bytes(res.content)
        coefficient_paths.append(str(filepath))
    return coefficient_paths


def _get_calibration(instrument: Instrument, args) -> dict:
    params = {
        "date": args.date,
        "instrumentPid": instrument.pid,
    }
    res = requests.get(
        f"{cloudnet_api_url}calibration",
        params=params,
        timeout=60,
    )
    if res.status_code == 404:
        return {}
    return res.json().get("data", {})


def _create_instrument_filepath(
    instrument: Instrument, args: argparse.Namespace
) -> str:
    folder = _create_output_folder("instrument", args)
    pid = _shorten_pid(instrument.pid)
    filename = f"{args.date.replace('-', '')}_{args.site}_{instrument.id}_{pid}.nc"
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


def _fetch_raw_meta(instruments: list[str], args: argparse.Namespace) -> list[dict]:
    res = requests.get(
        f"{cloudnet_api_url}raw-files/",
        params={
            "site": args.site,
            "date": args.date,
            "instrument": instruments,
            "status": ["uploaded", "processed"],
        },
        timeout=60,
    )
    res.raise_for_status()
    return res.json()


def _filter_by_instrument(meta: list[dict], instrument: Instrument) -> list[dict]:
    return [m for m in meta if m["instrumentInfo"]["pid"] == instrument.pid]


def _filter_by_suffix(meta: list[dict], product: str) -> list[dict]:
    if product == "radar":
        meta = [m for m in meta if not m["filename"].lower().endswith(".lv0")]
    elif product == "mwr":
        meta = [
            m for m in meta if re.search(r"\.(lwp|iwv)", m["filename"], re.IGNORECASE)
        ]
    elif product == "mwr-l1c":
        meta = [m for m in meta if not m["filename"].lower().endswith(".nc")]
    return meta


def _get_source_instruments(products: list[str]) -> dict[str, list[str]]:
    source_instruments = {}
    for product in products:
        prod, model = _parse_instrument(product)
        res = requests.get(f"{cloudnet_api_url}products/{prod}", timeout=60)
        res.raise_for_status()
        if sources := res.json().get("sourceInstruments", []):
            source_instruments[prod] = [i["id"] for i in sources]
            if match := [i for i in source_instruments[prod] if i == model]:
                source_instruments[prod] = match
    return source_instruments


def _get_product_sources(products: list[str]) -> dict[str, list[str]]:
    source_products = {}
    for product in products:
        prod, _ = _parse_instrument(product)
        res = requests.get(f"{cloudnet_api_url}products/{prod}", timeout=60)
        res.raise_for_status()
        if sources := res.json().get("sourceProducts", []):
            source_products[prod] = [i["id"] for i in sources]
    return source_products


def _parse_instrument(s: str) -> tuple[str, str | None]:
    if "[" in s and s.endswith("]"):
        name = s[: s.index("[")]
        value = s[s.index("[") + 1 : -1]
    else:
        name = s
        value = None
    return name, value


def _select_instrument(meta: list[dict], product: str) -> Instrument | None:
    instruments = _get_unique_instruments(meta)
    if len(instruments) == 0:
        logging.info("No instruments found")
        return None
    if len(instruments) > 1:
        logging.info("Multiple instruments found for %s", product)
        logging.info("Please specify which one to use")
        for i, instrument in enumerate(instruments):
            logging.info("%d: %s", i + 1, instrument.name)
        ind = int(input("Select: ")) - 1
        selected_instrument = instruments[ind]
    else:
        selected_instrument = instruments[0]
        logging.info("Single instrument found: %s", selected_instrument.name)
    return selected_instrument


def _get_unique_instruments(meta: list[dict]) -> list[Instrument]:
    unique_pids = {m["instrumentInfo"]["pid"] for m in meta}
    unique_instruments = []
    for pid in unique_pids:
        for m in meta:
            if m["instrumentInfo"]["pid"] == pid:
                i = m["instrumentInfo"]
                unique_instruments.append(
                    Instrument(i["instrumentId"], i["pid"], i["name"])
                )
                break
    return sorted(unique_instruments, key=lambda x: x.name)


def _fetch_product(
    args: argparse.Namespace, product: str, source: str | None = None
) -> str | None:
    payload = {
        "date": args.date,
        "site": args.site,
        "product": product,
    }
    url = f"{cloudnet_api_url}files"
    res = requests.get(url, payload, timeout=60)
    res.raise_for_status()
    meta = res.json()
    if source:
        meta = [
            m for m in meta if "instrument" in m and m["instrument"]["id"] == source
        ]
    if not meta:
        logging.info("No data available for %s", product)
        return None
    if len(meta) > 1:
        logging.info(
            "Multiple files for %s ... taking the first but some logic needed", product
        )
    meta = meta[0]
    suffix = "geophysical" if "geophysical" in meta["product"]["type"] else "instrument"
    folder = _create_output_folder(suffix, args)
    return _download_product_file(meta, folder)


def _fetch_model(args: argparse.Namespace) -> str | None:
    payload = {
        "date": args.date,
        "site": args.site,
    }
    url = f"{cloudnet_api_url}model-files"
    res = requests.get(url, payload, timeout=60)
    res.raise_for_status()
    meta = res.json()
    if not meta:
        logging.info("No model data available for this date")
        return None
    meta = meta[0]
    folder = _create_output_folder("instrument", args)
    return _download_product_file(meta, folder)


def _fetch_raw(metadata: list[dict], args: argparse.Namespace) -> list[str]:
    pid = _shorten_pid(metadata[0]["instrumentInfo"]["pid"])
    instrument = f"{metadata[0]['instrumentInfo']['instrumentId']}_{pid}"
    folder = _create_input_folder(instrument, args)
    filepaths = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_download_raw_file, meta, folder) for meta in metadata
        ]
        for future in as_completed(futures):
            filepaths.append(future.result())
    return filepaths


def _download_raw_file(meta: dict, folder: Path) -> str:
    filepath = folder / meta["filename"]
    possible_filepaths = [filepath]
    if filepath.suffix == ".gz":
        possible_filepaths.append(filepath.with_suffix(""))
    for path in possible_filepaths:
        if path.exists() and md5sum(path) == meta["checksum"]:
            logging.info("Existing file found: %s", path)
            return str(path)
    logging.info("Downloading file: %s", filepath)
    res = requests.get(meta["downloadUrl"], timeout=60)
    res.raise_for_status()
    filepath.write_bytes(res.content)
    if filepath.suffix == ".gz":
        filepath = _unzip_gz_file(filepath)
    return str(filepath)


def _download_product_file(meta: dict, folder: Path) -> str:
    filepath = folder / meta["filename"]
    if filepath.exists():
        logging.info("Existing file found: %s", filepath)
        return str(filepath)
    logging.info("Downloading file: %s", filepath)
    res = requests.get(meta["downloadUrl"], timeout=60)
    res.raise_for_status()
    filepath.write_bytes(res.content)
    return str(filepath)


def _unzip_gz_file(path_in: Path) -> Path:
    if path_in.suffix != ".gz":
        return path_in
    path_out = path_in.with_suffix("")
    logging.debug("Decompressing %s to %s", path_in, path_out)
    with gzip.open(path_in, "rb") as file_in, open(path_out, "wb") as file_out:
        shutil.copyfileobj(file_in, file_out)
    path_in.unlink()
    return path_out


def _read_site_meta(meta: list[dict]) -> dict:
    return {
        "latitude": meta[0]["site"]["latitude"],
        "longitude": meta[0]["site"]["longitude"],
        "altitude": meta[0]["site"]["altitude"],
        "name": meta[0]["site"]["humanReadableName"],
    }


def _shorten_pid(pid: str) -> str:
    return pid.split(".")[-1][:8]


def _check_input(files: list) -> None:
    if len(files) > 1:
        msg = "Multiple input files found"
        raise ValueError(msg)


def _plot(filepath: os.PathLike | str | None, product: str, args: argparse.Namespace):
    if filepath is None or (not args.plot and not args.show):
        return
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
    getattr(module, f"generate_{product.replace('-','_')}")(mwr_l1c_file, output_file)
    logging.info("Processed %s: %s", product, output_file)
    return str(output_file)


def _fetch_cloudnet_sites() -> list[str]:
    res = requests.get(f"{cloudnet_api_url}sites", timeout=60)
    res.raise_for_status()
    return [site["id"] for site in res.json()]


def _parse_products(product_argument: str) -> list[str]:
    products = product_argument.split(",")
    res = requests.get(f"{cloudnet_api_url}products", timeout=60)
    res.raise_for_status()
    valid_options = [p["id"] for p in res.json()]
    valid_products = []
    for product in products:
        prod, _ = _parse_instrument(product)
        if prod in valid_options:
            valid_products.append(product)
    return valid_products


def main():
    parser = argparse.ArgumentParser(
        description="Command line interface for running CloudnetPy."
    )
    parser.add_argument(
        "-s",
        "--site",
        type=str,
        help="Site",
        required=True,
        choices=_fetch_cloudnet_sites(),
        metavar="SITE",
    )
    parser.add_argument(
        "-d", "--date", type=str, help="Date in YYYY-MM-DD", required=True
    )
    parser.add_argument(
        "-p",
        "--products",
        type=_parse_products,
        help=(
            "Products to process, e.g. 'radar' or 'classification'. If the site "
            "has many instruments, you can specify the instrument in brackets, "
            "e.g. radar[mira-35]."
        ),
        required=True,
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
    args = parser.parse_args()

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.handlers = [handler]

    with TemporaryDirectory() as tmpdir:
        run(args, tmpdir)


if __name__ == "__main__":
    main()
