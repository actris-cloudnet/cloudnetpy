import datetime
import logging
import math
from collections.abc import Sequence
from os import PathLike
from uuid import UUID

import numpy as np
import numpy.typing as npt
from numpy import ma
from rpgpy import RPGFileError

from cloudnetpy import output, utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import G_TO_KG, HPA_TO_PA, KM_H_TO_M_S, MM_H_TO_M_S
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.instruments.instruments import Instrument
from cloudnetpy.instruments.rpg_reader import Fmcw94Bin, HatproBinCombined
from cloudnetpy.metadata import MetaData


def rpg2nc(
    path_to_l1_files: str | PathLike,
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> tuple[UUID, list[str]]:
    """Converts RPG-FMCW-94 cloud radar data into Cloudnet Level 1b netCDF file.

    This function reads one day of RPG Level 1 cloud radar binary files,
    concatenates the data and writes a netCDF file.

    Args:
        path_to_l1_files: Folder containing one day of RPG LV1 files.
        output_file: Output file name.
        site_meta: Dictionary containing information about the
            site. Required key value pairs are `altitude` (metres above mean
            sea level) and `name`.
        uuid: Set specific UUID for the file.
        date: Expected date in the input files. If not set,
            all files will be used. This might cause unexpected behavior if
            there are files from several days. If date is set as 'YYYY-MM-DD',
            only files that match the date will be used.

    Returns:
        2-element tuple containing

        - UUID of the generated file.
        - Files used in the processing.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    Examples:
        >>> from cloudnetpy.instruments import rpg2nc
        >>> site_meta = {'name': 'Hyytiala', 'altitude': 174}
        >>> rpg2nc('/path/to/files/', 'test.nc', site_meta)

    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = utils.get_uuid(uuid)
    l1_files = utils.get_sorted_filenames(path_to_l1_files, ".LV1")
    fmcw94_objects, valid_files = _get_fmcw94_objects(l1_files, date)
    one_day_of_data = create_one_day_data_record(fmcw94_objects)
    one_day_of_data["nyquist_velocity"] = _expand_nyquist(one_day_of_data)
    _print_info(one_day_of_data)
    fmcw = Fmcw(one_day_of_data, site_meta)
    fmcw.convert_time_to_fraction_hour()
    fmcw.mask_invalid_ldr()
    fmcw.mask_invalid_width()
    fmcw.sort_timestamps()
    fmcw.remove_duplicate_timestamps()
    fmcw.linear_to_db(("Zh", "antenna_gain"))
    fmcw.convert_units()
    fmcw.add_site_geolocation()
    valid_ind = fmcw.add_zenith_angle()
    fmcw.screen_time_indices(valid_ind)
    fmcw.add_height()
    if len(np.unique(fmcw.data["time"][:].astype("f4"))) != len(fmcw.data["time"][:]):
        msg = "Convert time to f8 to keep values unique in netCDF"
        logging.info(msg)
        fmcw.data["time"].data_type = "f8"
    attributes = output.add_time_attribute(RPG_ATTRIBUTES, fmcw.date)
    output.update_attributes(fmcw.data, attributes)
    output.save_level1b(fmcw, output_file, uuid)
    return uuid, valid_files


def _print_info(data: dict) -> None:
    dual_pol = data["dual_polarization"]
    if dual_pol == 0:
        mode = "single polarisation"
    elif dual_pol == 1:
        mode = "LDR"
    else:
        mode = "STSR"
    logging.info("RPG cloud radar in %s mode", mode)


RpgObjects = Sequence[Fmcw94Bin] | Sequence[HatproBinCombined]


def create_one_day_data_record(rpg_objects: RpgObjects) -> dict:
    """Concatenates all RPG FMCW / HATPRO data from one day."""
    rpg_raw_data, rpg_header = _stack_rpg_data(rpg_objects)
    if "range" in rpg_header:
        rpg_header["range"] = rpg_objects[0].header["range"]
    should_be_constant = [
        "customer_name",
        "model_number",
        "dual_polarization",
        "antenna_separation",
        "antenna_diameter",
        "antenna_gain",
        "half_power_beam_width",
        "radar_frequency",
    ]
    to_be_removed = ["customer_name"]
    for key in should_be_constant:
        if key not in rpg_header:
            continue
        unique_values = np.unique(rpg_header[key])
        if len(unique_values) > 1:
            msg = f"More than one value for {key} found: {unique_values}"
            raise ValueError(msg)
        if key in to_be_removed:
            del rpg_header[key]
        else:
            rpg_header[key] = unique_values[0]

    rpg_raw_data = _mask_invalid_data(rpg_raw_data)
    return {**rpg_header, **rpg_raw_data}


def _expand_nyquist(data: dict) -> npt.NDArray:
    """Expands Nyquist velocity from time X chirp => time X range."""
    nyquist_velocity = ma.array(data["nyquist_velocity"])
    chirp_start_indices = ma.array(data["chirp_start_indices"])
    n_time = chirp_start_indices.shape[0]
    n_range = len(data["range"])
    expanded_nyquist = np.empty((n_time, n_range))
    for t in range(n_time):
        starts = chirp_start_indices[t].compressed()
        v_nyq = nyquist_velocity[t].compressed()
        ends = np.r_[starts[1:], n_range]
        seg_lengths = ends - starts
        expanded_nyquist[t, :] = np.repeat(v_nyq, seg_lengths)
    return expanded_nyquist


def _stack_rpg_data(rpg_objects: RpgObjects) -> tuple[dict, dict]:
    data: dict = {}
    header: dict = {}
    for rpg in rpg_objects:
        for src, dst in ((rpg.data, data), (rpg.header, header)):
            for name, value in src.items():
                if name.startswith("_"):
                    continue
                arr = dst.get(name)
                fun = (
                    ma.concatenate
                    if any(isinstance(x, ma.MaskedArray) for x in (value, arr))
                    else np.concatenate
                )
                dst[name] = fun((arr, value)) if arr is not None else value
    return data, header


def _mask_invalid_data(data_in: dict) -> dict:
    """Masks zeros and other fill values from data."""
    data = data_in.copy()
    fill_values = (-999, 1e-10)
    extra_keys = ("air_temperature", "air_pressure")
    for name in data:
        if np.issubdtype(data[name].dtype, np.integer) or (
            data[name].ndim < 2 and name not in extra_keys
        ):
            continue
        data[name] = ma.masked_equal(data[name], 0)
        for value in fill_values:
            data[name][data[name] == value] = ma.masked
            ind = np.isclose(data[name], value)
            data[name][ind] = ma.masked
    return data


def _get_fmcw94_objects(
    files: list[str], expected_date: datetime.date | None
) -> tuple[list[Fmcw94Bin], list[str]]:
    """Creates a list of Rpg() objects from the file names."""
    objects = []
    valid_files = []
    for file in files:
        try:
            obj = Fmcw94Bin(file)
            if expected_date is not None:
                _validate_date(obj, expected_date)
        except (RPGFileError, TypeError, ValueError, IndexError) as err:
            logging.warning(err)
            continue
        objects.append(obj)
        valid_files.append(file)
    if not objects:
        msg = "No valid files found"
        raise ValidTimeStampError(msg)
    objects = _interpolate_to_common_height(objects)
    objects = _pad_chirp_related_fields(objects)
    objects = _expand_time_related_fields(objects)
    return objects, valid_files


def _interpolate_to_common_height(objects: list[Fmcw94Bin]) -> list[Fmcw94Bin]:
    range_arrays = [obj.header["range"] for obj in objects]
    if all(np.array_equal(range_arrays[0], r) for r in range_arrays[1:]):
        return objects
    # Use range with the highest range gate for interpolation
    target_range = max(range_arrays, key=lambda r: r[-1])
    for obj in objects:
        src_range = obj.header["range"]
        if np.array_equal(src_range, target_range):
            continue
        for key, arr in obj.data.items():
            if arr.ndim == 2 and arr.shape[1] == src_range.size:
                obj.data[key] = utils.interpolate_2D_along_y(
                    src_range, arr, target_range
                )
        _interpolate_chirp_start_indices(obj, target_range)
        obj.header["range"] = target_range
    return objects


def _interpolate_chirp_start_indices(obj: Fmcw94Bin, range_new: np.ndarray) -> None:
    range_orig = obj.header["range"]
    vals = range_orig[obj.header["chirp_start_indices"]]
    obj.header["chirp_start_indices"] = np.abs(range_new[:, None] - vals).argmin(axis=0)


def _pad_chirp_related_fields(objects: list[Fmcw94Bin]) -> list[Fmcw94Bin]:
    """Pads chirp-related header fields with masked values to have the same length."""
    chirp_lens = [len(obj.header["chirp_start_indices"]) for obj in objects]
    if all(chirp_lens[0] == length for length in chirp_lens[1:]):
        return objects
    max_chirp_len = max(chirp_lens)
    for obj in objects:
        n_chirps = len(obj.header["chirp_start_indices"])
        if n_chirps == max_chirp_len:
            continue
        for key, arr in obj.header.items():
            if not isinstance(arr, str) and arr.ndim == 1 and arr.size == n_chirps:
                pad_len = max_chirp_len - n_chirps
                masked_arr = ma.array(arr, dtype=arr.dtype)
                pad = ma.masked_all(pad_len, dtype=arr.dtype)
                obj.header[key] = ma.concatenate([masked_arr, pad])
    return objects


def _expand_time_related_fields(objects: list[Fmcw94Bin]) -> list[Fmcw94Bin]:
    for obj in objects:
        n_time = obj.data["time"].size
        for key in obj.header:
            if key in ("range", "time") or key.startswith("_"):
                continue
            arr = obj.header[key]
            # Handle outliers in latitude and longitude (e.g. Galati 2024-02-11):
            if key in ("latitude", "longitude"):
                arr = ma.median(arr)
            if utils.isscalar(arr):
                obj.header[key] = np.repeat(arr, n_time)
            else:
                obj.header[key] = np.tile(arr, (n_time, 1))
    return objects


def _validate_date(obj: Fmcw94Bin, expected_date: datetime.date) -> None:
    for t in obj.data["time"][:]:
        date = utils.seconds2date(t).date()
        if date != expected_date:
            msg = "Ignoring a file (time stamps not what expected)"
            raise ValueError(msg)


class Rpg(CloudnetInstrument):
    """Base class for RPG FMCW-94 cloud radar and HATPRO mwr."""

    def __init__(self, raw_data: dict, site_meta: dict) -> None:
        super().__init__()
        self.raw_data = raw_data
        self.site_meta = site_meta
        self.date = self._get_date()
        self.data = self._init_data()
        self.instrument: Instrument

    def convert_time_to_fraction_hour(self, data_type: str | None = None) -> None:
        """Converts time to fraction hour."""
        ms2s = 1e-3
        total_time_sec = self.raw_data["time"] + self.raw_data.get("time_ms", 0) * ms2s
        fraction_hour = utils.seconds2hours(total_time_sec)

        self.data["time"] = CloudnetArray(
            np.array(fraction_hour),
            "time",
            data_type=data_type,
        )

    def _get_date(self) -> datetime.date:
        time_first = self.raw_data["time"][0]
        time_last = self.raw_data["time"][-1]
        date_first = utils.seconds2date(time_first).date()
        date_last = utils.seconds2date(time_last).date()
        if date_first != date_last:
            logging.warning("Measurements from different days")
        return date_first

    def _init_data(self) -> dict:
        data = {}
        for key in self.raw_data:
            data[key] = CloudnetArray(self.raw_data[key], key)
        return data


class Fmcw(Rpg):
    """Class for RPG cloud radars."""

    def __init__(self, raw_data: dict, site_properties: dict) -> None:
        super().__init__(raw_data, site_properties)
        self.instrument = self._get_instrument(raw_data)

    def mask_invalid_ldr(self) -> None:
        """Removes ldr outliers."""
        threshold = -35
        if "ldr" in self.data:
            self.data["ldr"].data = ma.masked_less_equal(
                self.data["ldr"].data,
                threshold,
            )

    def mask_invalid_width(self) -> None:
        """Removes very low width values.

        Simplified method. Threshold value should depend on the radar
            settings and vary at each chirp.
        """
        threshold = 0.005
        ind = np.where(self.data["width"].data < threshold)
        for key in ("Zh", "v", "ldr", "width", "sldr"):
            if key in self.data:
                self.data[key].data[ind] = ma.masked

    def add_zenith_angle(self) -> list:
        """Adds zenith angle and returns time indices with valid zenith angle."""
        elevation = self.data["elevation"].data
        zenith = 90 - elevation
        is_valid_zenith = _filter_zenith_angle(zenith)
        n_removed = len(is_valid_zenith) - np.count_nonzero(is_valid_zenith)
        if n_removed == len(zenith):
            msg = "No profiles with valid zenith angle"
            raise ValidTimeStampError(msg)
        if n_removed > 0:
            logging.warning(
                "Filtering %s profiles due to invalid zenith angle",
                n_removed,
            )
        self.data["zenith_angle"] = CloudnetArray(zenith, "zenith_angle")
        del self.data["elevation"]
        return list(is_valid_zenith)

    def convert_units(self) -> None:
        """Converts units."""
        self.data["rainfall_rate"].data = self.data["rainfall_rate"].data * MM_H_TO_M_S
        self.data["lwp"].data *= G_TO_KG
        self.data["relative_humidity"].data /= 100
        self.data["air_pressure"].data *= HPA_TO_PA
        self.data["wind_speed"].data *= KM_H_TO_M_S

    @staticmethod
    def _get_instrument(data: dict) -> Instrument:
        frequency = data["radar_frequency"]
        if math.isclose(frequency, 35, abs_tol=0.1):
            return instruments.FMCW35
        if math.isclose(frequency, 94, abs_tol=0.1):
            return instruments.FMCW94
        msg = f"Unknown RPG cloud radar frequency: {frequency}"
        raise RuntimeError(msg)


class Hatpro(Rpg):
    """Class for RPG HATPRO mwr."""

    def __init__(
        self, raw_data: dict, site_properties: dict, instrument: Instrument
    ) -> None:
        super().__init__(raw_data, site_properties)
        self.instrument = instrument


def _filter_zenith_angle(zenith: ma.MaskedArray) -> npt.NDArray:
    """Returns indices of profiles with stable zenith angle close to 0 deg."""
    zenith = ma.array(zenith)
    if zenith.mask.all():
        return np.zeros(zenith.shape, dtype=bool)
    limits = [-5, 15]
    ind_close_to_zenith = np.where(
        np.logical_and(zenith > limits[0], zenith < limits[1]),
    )
    if not ind_close_to_zenith[0].size:
        return np.zeros_like(zenith, dtype=bool)
    valid_range_median = ma.median(zenith[ind_close_to_zenith])
    is_stable_zenith = np.isclose(zenith, valid_range_median, atol=0.1)
    is_stable_zenith[zenith.mask] = False
    return np.array(is_stable_zenith)


DEFINITIONS = {
    "model_number": utils.status_field_definition(
        {
            0: "Single polarisation radar.",
            1: "Dual polarisation radar.",
        }
    ),
    "dual_polarization": utils.status_field_definition(
        {
            0: """Single polarisation radar.""",
            1: """Dual polarisation radar in linear depolarisation ratio (LDR)
                  mode.""",
            2: """Dual polarisation radar in simultaneous transmission
                  simultaneous reception (STSR) mode.""",
        }
    ),
    "FFT_window": utils.status_field_definition(
        {
            0: "Square",
            1: "Parzen",
            2: "Blackman",
            3: "Welch",
            4: "Slepian2",
            5: "Slepian3",
        }
    ),
    "quality_flag": utils.bit_field_definition(
        {
            0: "ADC saturation.",
            1: "Spectral width too high.",
            2: "No transmission power levelling.",
        }
    ),
}

RPG_ATTRIBUTES = {
    # LDR-mode radars:
    "ldr": MetaData(
        long_name="Linear depolarisation ratio",
        units="dB",
        dimensions=("time", "range"),
    ),
    "rho_cx": MetaData(
        long_name="Co-cross-channel correlation coefficient",
        units="1",
        dimensions=("time", "range"),
    ),
    "phi_cx": MetaData(
        long_name="Co-cross-channel differential phase",
        units="rad",
        dimensions=("time", "range"),
    ),
    # STSR-mode radars
    "zdr": MetaData(
        long_name="Differential reflectivity", units="dB", dimensions=("time", "range")
    ),
    "rho_hv": MetaData(
        long_name="Correlation coefficient", units="1", dimensions=("time", "range")
    ),
    "phi_dp": MetaData(
        long_name="Differential phase", units="rad", dimensions=("time", "range")
    ),
    "srho_hv": MetaData(
        long_name="Slanted correlation coefficient",
        units="1",
        dimensions=("time", "range"),
    ),
    "kdp": MetaData(
        long_name="Specific differential phase shift",
        units="rad km-1",
        dimensions=("time", "range"),
    ),
    "differential_attenuation": MetaData(
        long_name="Differential attenuation",
        units="dB km-1",
        dimensions=("time", "range"),
    ),
    # All radars
    "file_code": MetaData(
        long_name="File code",
        units="1",
        comment="Indicates the RPG software version.",
        dimensions=("time",),
    ),
    "program_number": MetaData(
        long_name="Program number", units="1", dimensions=("time",)
    ),
    "model_number": MetaData(
        long_name="Model number",
        units="1",
        definition=DEFINITIONS["model_number"],
        dimensions=None,
    ),
    "antenna_separation": MetaData(
        long_name="Antenna separation", units="m", dimensions=None
    ),
    "antenna_diameter": MetaData(
        long_name="Antenna diameter", units="m", dimensions=None
    ),
    "antenna_gain": MetaData(long_name="Antenna gain", units="dB", dimensions=None),
    "half_power_beam_width": MetaData(
        long_name="Half power beam width", units="degree", dimensions=None
    ),
    "dual_polarization": MetaData(
        long_name="Dual polarisation type",
        units="1",
        definition=DEFINITIONS["dual_polarization"],
        dimensions=None,
    ),
    "sample_duration": MetaData(
        long_name="Sample duration", units="s", dimensions=("time",)
    ),
    "calibration_interval": MetaData(
        long_name="Calibration interval in samples", units="1", dimensions=("time",)
    ),
    "number_of_spectral_samples": MetaData(
        long_name="Number of spectral samples in each chirp sequence",
        units="1",
        dimensions=("time", "chirp_sequence"),
    ),
    "number_of_averaged_chirps": MetaData(
        long_name="Number of averaged chirps in sequence",
        units="1",
        dimensions=("time", "chirp_sequence"),
    ),
    "chirp_start_indices": MetaData(
        long_name="Chirp sequences start indices",
        units="1",
        dimensions=("time", "chirp_sequence"),
    ),
    "integration_time": MetaData(
        long_name="Integration time",
        units="s",
        comment="Effective integration time of chirp sequence",
        dimensions=("time", "chirp_sequence"),
    ),
    "range_resolution": MetaData(
        long_name="Vertical resolution of range",
        units="m",
        dimensions=("time", "chirp_sequence"),
    ),
    "FFT_window": MetaData(
        long_name="FFT window type",
        units="1",
        definition=DEFINITIONS["FFT_window"],
        dimensions=("time",),
    ),
    "input_voltage_range": MetaData(
        long_name="ADC input voltage range (+/-)", units="mV", dimensions=("time",)
    ),
    "noise_threshold": MetaData(
        long_name="Noise filter threshold factor",
        units="1",
        comment="Multiple of the standard deviation of Doppler spectra.",
        dimensions=("time",),
    ),
    "time_ms": MetaData(long_name="Time ms", units="ms", dimensions=("time",)),
    "quality_flag": MetaData(
        long_name="Quality flag",
        definition=DEFINITIONS["quality_flag"],
        units="1",
        dimensions=("time",),
    ),
    "voltage": MetaData(long_name="Voltage", units="V", dimensions=("time",)),
    "brightness_temperature": MetaData(
        long_name="Brightness temperature", units="K", dimensions=("time",)
    ),
    "if_power": MetaData(long_name="IF power at ACD", units="uW", dimensions=("time",)),
    "status_flag": MetaData(
        long_name="Status flag for heater and blower", units="1", dimensions=("time",)
    ),
    "transmitted_power": MetaData(
        long_name="Transmitted power", units="W", dimensions=("time",)
    ),
    "transmitter_temperature": MetaData(
        long_name="Transmitter temperature", units="K", dimensions=("time",)
    ),
    "receiver_temperature": MetaData(
        long_name="Receiver temperature", units="K", dimensions=("time",)
    ),
    "pc_temperature": MetaData(
        long_name="PC temperature", units="K", dimensions=("time",)
    ),
}
