"""This module contains RPG Cloud Radar related functions."""

import logging
import math
from collections.abc import Sequence

import numpy as np
from numpy import ma
from rpgpy import RPGFileError

from cloudnetpy import output, utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.constants import G_TO_KG, HPA_TO_PA, KM_H_TO_M_S, MM_H_TO_M_S
from cloudnetpy.exceptions import InconsistentDataError, ValidTimeStampError
from cloudnetpy.instruments import instruments
from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument
from cloudnetpy.instruments.instruments import Instrument
from cloudnetpy.instruments.rpg_reader import Fmcw94Bin, HatproBinCombined
from cloudnetpy.metadata import MetaData


def rpg2nc(
    path_to_l1_files: str,
    output_file: str,
    site_meta: dict,
    uuid: str | None = None,
    date: str | None = None,
) -> tuple[str, list]:
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
    l1_files = utils.get_sorted_filenames(path_to_l1_files, ".LV1")
    fmcw94_objects, valid_files = _get_fmcw94_objects(l1_files, date)
    one_day_of_data = create_one_day_data_record(fmcw94_objects)
    if not valid_files:
        return "", []
    print_info(one_day_of_data)
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
    attributes = output.add_time_attribute(RPG_ATTRIBUTES, fmcw.date)
    output.update_attributes(fmcw.data, attributes)
    uuid = output.save_level1b(fmcw, output_file, uuid)
    return uuid, valid_files


def print_info(data: dict) -> None:
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
    """Concatenates all RPG data from one day."""
    rpg_raw_data, rpg_header = _stack_rpg_data(rpg_objects)
    if len(rpg_objects) > 1:
        rpg_header = _reduce_header(rpg_header)
    rpg_raw_data = _mask_invalid_data(rpg_raw_data)
    return {**rpg_header, **rpg_raw_data}


def _stack_rpg_data(rpg_objects: RpgObjects) -> tuple[dict, dict]:
    """Combines data from hourly RPG objects.

    Notes:
        Ignores variable names starting with an underscore.

    """

    def _stack(source, target, fun) -> None:
        for name, value in source.items():
            if not name.startswith("_"):
                target[name] = fun((target[name], value)) if name in target else value

    data: dict = {}
    header: dict = {}
    for rpg in rpg_objects:
        _stack(rpg.data, data, ma.concatenate)
        _stack(rpg.header, header, ma.vstack)
    return data, header


def _reduce_header(header: dict) -> dict:
    """Removes duplicate header data. Otherwise, we would need n_files dimension."""
    reduced_header = {}
    for key, data in header.items():
        first_profile_value = data[0]
        is_identical_value = bool(
            np.isclose(data, first_profile_value, rtol=1e-2).all(),
        )
        if is_identical_value is False:
            msg = f"Inconsistent header: {key}: {data}"
            if key in (
                "latitude",
                "longitude",
                "sample_duration",
                "calibration_interval",
                "noise_threshold",
            ):
                logging.warning(msg)
            else:
                raise InconsistentDataError(msg)
        reduced_header[key] = first_profile_value
    return reduced_header


def _mask_invalid_data(data_in: dict) -> dict:
    """Masks zeros and other fill values from data."""
    data = data_in.copy()
    fill_values = (-999, 1e-10)
    extra_keys = ("air_temperature", "air_pressure")
    for name in data:
        if (
            np.issubdtype(data[name].dtype, np.integer)
            or data[name].ndim < 2
            and name not in extra_keys
        ):
            continue
        data[name] = ma.masked_equal(data[name], 0)
        for value in fill_values:
            data[name][data[name] == value] = ma.masked
            ind = np.isclose(data[name], value)
            data[name][ind] = ma.masked
    return data


def _get_fmcw94_objects(files: list, expected_date: str | None) -> tuple[list, list]:
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
    if objects:
        objects, valid_files = _remove_files_with_bad_height(objects, valid_files)
    if not valid_files:
        raise ValidTimeStampError
    return objects, valid_files


def _remove_files_with_bad_height(objects: list, files: list) -> tuple[list, list]:
    lengths = [obj.data["Zh"].shape[1] for obj in objects]
    most_common = np.bincount(lengths).argmax()
    files = [
        file
        for file, obj, length in zip(files, objects, lengths, strict=True)
        if length == most_common
    ]
    objects = [
        obj
        for obj, length in zip(objects, lengths, strict=True)
        if length == most_common
    ]
    n_removed = len(lengths) - len(files)
    if n_removed > 0:
        logging.warning(
            "Removed %s RPG-FMCW-94 files due to inconsistent height vector",
            n_removed,
        )
    return objects, files


def _validate_date(obj, expected_date: str) -> None:
    for t in obj.data["time"][:]:
        date_str = "-".join(utils.seconds2date(t)[:3])
        if date_str != expected_date:
            msg = "Ignoring a file (time stamps not what expected)"
            raise ValueError(msg)


class Rpg(CloudnetInstrument):
    """Base class for RPG FMCW-94 cloud radar and HATPRO mwr."""

    def __init__(self, raw_data: dict, site_meta: dict):
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

    def _get_date(self) -> list:
        time_first = self.raw_data["time"][0]
        time_last = self.raw_data["time"][-1]
        date_first = utils.seconds2date(time_first)[:3]
        date_last = utils.seconds2date(time_last)[:3]
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

    def __init__(self, raw_data: dict, site_properties: dict):
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
    def _get_instrument(data: dict):
        frequency = data["radar_frequency"]
        if math.isclose(frequency, 35, abs_tol=0.1):
            return instruments.FMCW35
        if math.isclose(frequency, 94, abs_tol=0.1):
            return instruments.FMCW94
        msg = f"Unknown RPG cloud radar frequency: {frequency}"
        raise RuntimeError(msg)


class Hatpro(Rpg):
    """Class for RPG HATPRO mwr."""

    def __init__(self, raw_data: dict, site_properties: dict, instrument: Instrument):
        super().__init__(raw_data, site_properties)
        self.instrument = instrument


def _filter_zenith_angle(zenith: ma.MaskedArray) -> np.ndarray:
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
    "ldr": MetaData(long_name="Linear depolarisation ratio", units="dB"),
    "rho_cx": MetaData(long_name="Co-cross-channel correlation coefficient", units="1"),
    "phi_cx": MetaData(long_name="Co-cross-channel differential phase", units="rad"),
    # STSR-mode radars
    "zdr": MetaData(long_name="Differential reflectivity", units="dB"),
    "rho_hv": MetaData(long_name="Correlation coefficient", units="1"),
    "phi_dp": MetaData(long_name="Differential phase", units="rad"),
    "srho_hv": MetaData(long_name="Slanted correlation coefficient", units="1"),
    "kdp": MetaData(long_name="Specific differential phase shift", units="rad km-1"),
    "differential_attenuation": MetaData(
        long_name="Differential attenuation",
        units="dB km-1",
    ),
    # All radars
    "file_code": MetaData(
        long_name="File code",
        units="1",
        comment="Indicates the RPG software version.",
    ),
    "program_number": MetaData(long_name="Program number", units="1"),
    "model_number": MetaData(
        long_name="Model number",
        units="1",
        definition=DEFINITIONS["model_number"],
    ),
    "antenna_separation": MetaData(
        long_name="Antenna separation",
        units="m",
    ),
    "antenna_diameter": MetaData(
        long_name="Antenna diameter",
        units="m",
    ),
    "antenna_gain": MetaData(
        long_name="Antenna gain",
        units="dB",
    ),
    "half_power_beam_width": MetaData(
        long_name="Half power beam width",
        units="degree",
    ),
    "dual_polarization": MetaData(
        long_name="Dual polarisation type",
        units="1",
        definition=DEFINITIONS["dual_polarization"],
    ),
    "sample_duration": MetaData(long_name="Sample duration", units="s"),
    "calibration_interval": MetaData(
        long_name="Calibration interval in samples",
        units="1",
    ),
    "number_of_spectral_samples": MetaData(
        long_name="Number of spectral samples in each chirp sequence",
        units="1",
    ),
    "number_of_averaged_chirps": MetaData(
        long_name="Number of averaged chirps in sequence",
        units="1",
    ),
    "integration_time": MetaData(
        long_name="Integration time",
        units="s",
        comment="Effective integration time of chirp sequence",
    ),
    "range_resolution": MetaData(
        long_name="Vertical resolution of range",
        units="m",
    ),
    "FFT_window": MetaData(
        long_name="FFT window type",
        units="1",
        definition=DEFINITIONS["FFT_window"],
    ),
    "input_voltage_range": MetaData(
        long_name="ADC input voltage range (+/-)",
        units="mV",
    ),
    "noise_threshold": MetaData(
        long_name="Noise filter threshold factor",
        units="1",
        comment="Multiple of the standard deviation of Doppler spectra.",
    ),
    "time_ms": MetaData(
        long_name="Time ms",
        units="ms",
    ),
    "quality_flag": MetaData(
        long_name="Quality flag",
        definition=DEFINITIONS["quality_flag"],
        units="1",
    ),
    "voltage": MetaData(
        long_name="Voltage",
        units="V",
    ),
    "brightness_temperature": MetaData(
        long_name="Brightness temperature",
        units="K",
    ),
    "if_power": MetaData(
        long_name="IF power at ACD",
        units="uW",
    ),
    "status_flag": MetaData(
        long_name="Status flag for heater and blower",
        units="1",
    ),
    "transmitted_power": MetaData(
        long_name="Transmitted power",
        units="W",
    ),
    "transmitter_temperature": MetaData(
        long_name="Transmitter temperature",
        units="K",
    ),
    "receiver_temperature": MetaData(
        long_name="Receiver temperature",
        units="K",
    ),
    "pc_temperature": MetaData(
        long_name="PC temperature",
        units="K",
    ),
    "kurtosis": MetaData(
        long_name="Kurtosis of spectra",
        units="1",
    ),
    "correlation_coefficient": MetaData(
        long_name="Correlation coefficient",
        units="1",
    ),
}
