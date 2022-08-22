"""This module contains RPG Cloud Radar related functions."""
import logging
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from numpy import ma

from cloudnetpy import CloudnetArray, output, utils
from cloudnetpy.exceptions import InconsistentDataError, ValidTimeStampError
from cloudnetpy.instruments import general, instruments
from cloudnetpy.instruments.instruments import Instrument
from cloudnetpy.instruments.rpg_reader import Fmcw94Bin, HatproBinCombined
from cloudnetpy.metadata import MetaData


def rpg2nc(
    path_to_l1_files: str,
    output_file: str,
    site_meta: dict,
    uuid: Optional[str] = None,
    date: Optional[str] = None,
) -> Tuple[str, list]:
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
    general.linear_to_db(fmcw, ("Zh", "antenna_gain"))
    general.add_site_geolocation(fmcw)
    fmcw.add_zenith_angle()
    general.add_height(fmcw)
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
    logging.info(f"RPG cloud radar in {mode} mode")


RpgObjects = Union[Sequence[Fmcw94Bin], Sequence[HatproBinCombined]]


def create_one_day_data_record(rpg_objects: RpgObjects) -> dict:
    """Concatenates all RPG data from one day."""
    rpg_raw_data, rpg_header = _stack_rpg_data(rpg_objects)
    if len(rpg_objects) > 1:
        rpg_header = _reduce_header(rpg_header)
    rpg_raw_data = _mask_invalid_data(rpg_raw_data)
    return {**rpg_header, **rpg_raw_data}


def _stack_rpg_data(rpg_objects: RpgObjects) -> Tuple[dict, dict]:
    """Combines data from hourly RPG objects.

    Notes:
        Ignores variable names starting with an underscore.

    """

    def _stack(source, target, fun):
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
    """Removes duplicate header data."""
    reduced_header = {}
    for key, data in header.items():
        first_profile_value = data[0]
        is_identical_value = bool(np.isclose(data, first_profile_value, rtol=1e-2).all())
        if is_identical_value is False:
            msg = f"Inconsistent header: {key}"
            if key in ("latitude", "longitude"):
                logging.warning(msg)
            else:
                raise InconsistentDataError(msg)
        reduced_header[key] = first_profile_value
    return reduced_header


def _mask_invalid_data(data_in: dict) -> dict:
    """Masks zeros and other fill values from data."""
    data = data_in.copy()
    fill_values = (-999, 1e-10)
    for name in data:
        data[name] = ma.masked_equal(data[name], 0)
        for value in fill_values:
            data[name][data[name] == value] = ma.masked
            ind = np.isclose(data[name], value)
            data[name][ind] = ma.masked
    return data


def _get_fmcw94_objects(files: list, expected_date: Union[str, None]) -> Tuple[list, list]:
    """Creates a list of Rpg() objects from the file names."""
    objects = []
    valid_files = []
    for file in files:
        try:
            obj = Fmcw94Bin(file)
            if expected_date is not None:
                _validate_date(obj, expected_date)
        except (TypeError, ValueError) as err:
            logging.warning(err)
            continue
        objects.append(obj)
        valid_files.append(file)
    if objects:
        objects, valid_files = _remove_files_with_bad_height(objects, valid_files)
    if not valid_files:
        raise ValidTimeStampError
    return objects, valid_files


def _remove_files_with_bad_height(objects: list, files: list) -> Tuple[list, list]:
    lengths = [obj.data["Zh"].shape[1] for obj in objects]
    most_common = np.bincount(lengths).argmax()
    files = [file for file, obj, length in zip(files, objects, lengths) if length == most_common]
    objects = [obj for obj, length in zip(objects, lengths) if length == most_common]
    n_removed = len(lengths) - len(files)
    if n_removed > 0:
        logging.warning(f"Removed {n_removed} RPG-FMCW-94 files due to inconsistent height vector")
    return objects, files


def _validate_date(obj, expected_date: str) -> None:
    for t in obj.data["time"][:]:
        date_str = "-".join(utils.seconds2date(t)[:3])
        if date_str != expected_date:
            raise ValueError("Ignoring a file (time stamps not what expected)")


class Rpg:
    """Base class for RPG FMCW-94 cloud radar and HATPRO mwr."""

    def __init__(self, raw_data: dict, site_meta: dict):
        self.raw_data = raw_data
        self.site_meta = site_meta
        self.date = self._get_date()
        self.data = self._init_data()
        self.instrument: Instrument

    def convert_time_to_fraction_hour(self, data_type: Optional[str] = None) -> None:
        """Converts time to fraction hour."""
        key = "time"
        fraction_hour = utils.seconds2hours(self.raw_data[key])
        self.data[key] = CloudnetArray(np.array(fraction_hour), key, data_type=data_type)

    def sort_timestamps(self):
        """Sorts timestamps."""
        time = self.data["time"].data[:]
        ind = time.argsort()
        self._screen(ind)

    def remove_duplicate_timestamps(self):
        """Removes duplicate timestamps."""
        time = self.data["time"].data[:]
        _, ind = np.unique(time, return_index=True)
        self._screen(ind)

    def _screen(self, ind: np.ndarray):
        n_time = len(self.data["time"].data)
        for array in self.data.values():
            data = array.data
            if data.ndim > 0 and data.shape[0] == n_time:
                if data.ndim == 1:
                    screened_data = data[ind]
                else:
                    screened_data = data[ind, :]
                array.data = screened_data

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
            self.data["ldr"].data = ma.masked_less_equal(self.data["ldr"].data, threshold)

    def mask_invalid_width(self) -> None:
        """Removes very low width values.

        Simplified method. Threshold value should depend on the radar settings and vary at each
        chirp.
        """
        threshold = 0.005
        ind = np.where(self.data["width"].data < threshold)
        for key in ("Zh", "v", "ldr", "width", "sldr"):
            if key in self.data:
                self.data[key].data[ind] = ma.masked

    def add_zenith_angle(self) -> list:
        """Adds zenith angle and returns time indices where zenith angle is stable."""
        elevation = self.data["elevation"].data
        zenith = 90 - elevation
        if elevation.mask.all():
            zenith[:] = 0
            logging.warning("Can not determine zenith angle, assuming 0 degrees")
        is_stable_zenith = np.isclose(zenith, ma.median(zenith), atol=0.1)
        n_removed = len(is_stable_zenith) - np.count_nonzero(is_stable_zenith)
        if n_removed > 0:
            logging.warning(f"Filtering {n_removed} profiles due to varying zenith angle")
        self.data["zenith_angle"] = CloudnetArray(zenith, "zenith_angle")
        del self.data["elevation"]
        return list(is_stable_zenith)

    @staticmethod
    def _get_instrument(data: dict):
        frequency = data["radar_frequency"]
        if math.isclose(frequency, 35, abs_tol=0.1):
            return instruments.FMCW35
        if math.isclose(frequency, 94, abs_tol=0.1):
            return instruments.FMCW94
        raise RuntimeError(f"Unknown RPG cloud radar frequency: {frequency}")


class Hatpro(Rpg):
    """Class for RPG HATPRO mwr."""

    def __init__(self, raw_data: dict, site_properties: dict):
        super().__init__(raw_data, site_properties)
        self.instrument = instruments.HATPRO


DEFINITIONS = {
    "model_number": "\n" "0: Single polarisation radar.\n" "1: Dual polarisation radar.",
    "dual_polarization": (
        "\n"
        "Value 0: Single polarisation radar.\n"
        "Value 1: Dual polarisation radar in linear depolarisation ratio (LDR) mode.\n"
        "Value 2: Dual polarisation radar in simultaneous transmission simultaneous\n"
        "reception (STSR) mode."
    ),
    "FFT_window": (
        "\n"
        "Value 0: Square\n"
        "Value 1: Parzen\n"
        "Value 2: Blackman\n"
        "Value 3: Welch\n"
        "Value 4: Slepian2\n"
        "Value 5: Slepian3"
    ),
    "quality_flag": (
        "\n"
        "Bit 0: ADC saturation.\n"
        "Bit 1: Spectral width too high.\n"
        "Bit 2: No transmission power levelling."
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
    "sldr": MetaData(long_name="Slanted linear depolarisation ratio", units="dB"),
    "srho_hv": MetaData(long_name="Slanted correlation coefficient", units="1"),
    "kdp": MetaData(long_name="Specific differential phase shift", units="rad km-1"),
    "differential_attenuation": MetaData(long_name="Differential attenuation", units="dB km-1"),
    # All radars
    "file_code": MetaData(
        long_name="File code",
        units="1",
        comment="Indicates the RPG software version.",
    ),
    "program_number": MetaData(long_name="Program number", units="1"),
    "model_number": MetaData(
        long_name="Model number", units="1", definition=DEFINITIONS["model_number"]
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
        units="degrees",
    ),
    "dual_polarization": MetaData(
        long_name="Dual polarisation type", units="1", definition=DEFINITIONS["dual_polarization"]
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
    "chirp_start_indices": MetaData(
        long_name="Chirp sequences start indices",
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
        long_name="FFT window type", units="1", definition=DEFINITIONS["FFT_window"]
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
    "skewness": MetaData(
        long_name="Skewness of spectra",
        units="1",
    ),
    "kurtosis": MetaData(
        long_name="Kurtosis of spectra",
        units="1",
    ),
    "correlation_coefficient": MetaData(
        long_name="Correlation coefficient",
        units="1",
    ),
    "wind_direction": MetaData(
        long_name="Wind direction",
        units="degrees",
    ),
    "wind_speed": MetaData(
        long_name="Wind speed",
        units="m s-1",
    ),
    "relative_humidity": MetaData(
        long_name="Relative humidity",
        units="%",
    ),
    "rain_rate": MetaData(
        long_name="Rain rate",
        units="mm h-1",
    ),
}
