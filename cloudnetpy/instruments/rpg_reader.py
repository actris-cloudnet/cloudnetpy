import logging
from typing import BinaryIO, Literal

import numpy as np
from numpy import ma
from numpy.lib import recfunctions as rfn
from rpgpy import read_rpg

from cloudnetpy.constants import G_TO_KG
from cloudnetpy.exceptions import ValidTimeStampError


class Fmcw94Bin:
    """RPG Cloud Radar Level 1 data reader."""

    def __init__(self, filename):
        self.filename = filename
        self.header, self.data = read_rpg(filename)

        is_strs_mode = self.header.get("DualPol") == 2

        header_keymap = {
            "StartTime": "_start_time",
            "StopTime": "_stop_time",
            "SupPowLev": "_is_power_levelling",
            "SpkFilEna": "_is_spike_filter",
            "PhaseCorr": "_is_phase_correction",
            "RelPowCorr": "_is_relative_power_correction",
            "FFTWindow": "FFT_window",
            "FFTInputRng": "input_voltage_range",
            "SWVersion": "_software_version",
            "NoiseFilt": "noise_threshold",
            "FileCode": "file_code",
            "HeaderLen": "_HeaderLen",
            "CGProg": "program_number",
            "ModelNo": "model_number",
            "ProgName": "_program_name",
            "CustName": "_customer_name",
            "Freq": "radar_frequency",
            "AntSep": "antenna_separation",
            "AntDia": "antenna_diameter",
            "AntG": "antenna_gain",
            "HPBW": "half_power_beam_width",
            "DualPol": "dual_polarization",
            "SampDur": "sample_duration",
            "GPSLat": "latitude",
            "GPSLong": "longitude",
            "CalInt": "calibration_interval",
            "RAltN": "_number_of_range_gates",
            "TAltN": "_number_of_temperature_levels",
            "HAltN": "_number_of_humidity_levels",
            "SequN": "_number_of_chirp_sequences",
            "RAlts": "range",
            "TAlts": "_temperature_levels",
            "HAlts": "_humidity_levels",
            "SpecN": "number_of_spectral_samples",
            "RngOffs": "chirp_start_indices",
            "ChirpReps": "number_of_averaged_chirps",
            "SeqIntTime": "integration_time",
            "dR": "range_resolution",
            "MaxVel": "nyquist_velocity",
            "InstCalPar": "_calibration_period",
        }

        data_keymap = {
            "RefRat": "zdr" if is_strs_mode else "ldr",
            "CorrCoeff": "rho_hv" if is_strs_mode else "rho_cx",
            "DiffPh": "phi_dp" if is_strs_mode else "phi_cx",
            "Time": "time",
            "MSec": "time_ms",
            "QF": "quality_flag",
            "RR": "rainfall_rate",
            "RelHum": "relative_humidity",
            "EnvTemp": "air_temperature",
            "BaroP": "air_pressure",
            "WS": "wind_speed",
            "WD": "wind_direction",
            "DDVolt": "voltage",
            "DDTb": "brightness_temperature",
            "LWP": "lwp",
            "PowIF": "if_power",
            "Elev": "elevation",
            "Azi": "azimuth_angle",
            "Status": "status_flag",
            "TransPow": "transmitted_power",
            "TransT": "transmitter_temperature",
            "RecT": "receiver_temperature",
            "PCT": "pc_temperature",
            "Ze": "Zh",
            "MeanVel": "v",
            "SpecWidth": "width",
            "Skewn": "skewness",
            "Kurt": "kurtosis",
            "SLDR": "sldr",
            "SCorrCoeff": "srho_hv",
            "KDP": "kdp",
            "DiffAtt": "differential_attenuation",
        }

        self.replace_keys(self.header, header_keymap)
        self.replace_keys(self.data, data_keymap)

    @staticmethod
    def replace_keys(d: dict, keymap: dict):
        for key in d.copy():
            if key in keymap:
                new_key = keymap[key]
                d[new_key] = d.pop(key)


def _read_from_file(
    file: BinaryIO,
    fields: list[tuple[str, str]],
    count: int | None = None,
) -> ma.MaskedArray:
    arr = np.fromfile(file, np.dtype(fields), 1 if count is None else count)
    masked_arr = ma.array(arr)
    if count is None:
        return masked_arr[0]
    return masked_arr


def _decode_angles(
    x: np.ndarray,
    version: Literal[1, 2],
) -> tuple[np.ndarray, np.ndarray]:
    """Decode elevation and azimuth angles.

    >>> _decode_angles(np.array([1267438.5]), version=1)
    (array([138.5]), array([267.4]))
    >>> _decode_angles(np.array([1453031045, -900001232]), version=2)
    (array([145.3, -90. ]), array([310.45,  12.32]))

    Based on `interpret_angle` from mwr_raw2l1 licensed under BSD 3-Clause:
    https://github.com/MeteoSwiss/mwr_raw2l1/blob/0738490d22f77138cdf9329bf102f319c78be584/mwr_raw2l1/readers/reader_rpg_helpers.py#L30
    """
    if version == 1:
        # Description in the manual is quite unclear so here's an improved one:
        # Ang=sign(El)*(|El|+1000*Az), -90°<=El<100°, 0°<=Az<360°. If El>=100°
        # (i.e. requires 3 digits), the value 1000.000 is added to Ang and El in
        # the formula is El-100°. For decoding to make sense, Az and El must be
        # stored in precision of 0.1 degrees.

        ele_offset = np.zeros(x.shape)
        ind_offset_corr = x >= 1e6
        ele_offset[ind_offset_corr] = 100
        x = np.copy(x)
        x[ind_offset_corr] -= 1e6

        azi = (np.abs(x) // 100) / 10
        ele = x - np.sign(x) * azi * 1000 + ele_offset
    elif version == 2:
        # First 5 decimal digits is azimuth*100, last 5 decimal digits is
        # elevation*100, sign of Ang is sign of elevation.
        ele = np.sign(x) * (np.abs(x) // 1e5) / 100
        azi = (np.abs(x) - np.abs(ele) * 1e7) / 100
    else:
        msg = f"Known versions for angle encoding are 1 and 2, but received {version}"
        raise NotImplementedError(msg)

    return ele, azi


class HatproBin:
    """HATPRO binary file reader. Byte order is assumed to be little endian.

    References:
        Radiometer Physics (2014): Instrument Operation and Software Guide
        Operation Principles and Software Description for RPG standard single
        polarization radiometers (G5 series).
        https://www.radiometer-physics.de/download/PDF/Radiometers/HATPRO/RPG_MWR_STD_Software_Manual%20G5.pdf
    """

    header: ma.MaskedArray
    data: ma.MaskedArray
    version: Literal[1, 2]
    variable: str

    QUALITY_NA = 0
    QUALITY_HIGH = 1
    QUALITY_MEDIUM = 2
    QUALITY_LOW = 3

    def __init__(self, filename):
        self.filename = filename
        with open(self.filename, "rb") as file:
            self._read_header(file)
            self._read_data(file)
        self._remove_duplicate_timestamps()
        self._add_zenith_angle()

    def screen_bad_profiles(self) -> None:
        is_bad = self.data["_quality_flag"] & 0b110 == self.QUALITY_LOW << 1
        n_bad = np.count_nonzero(is_bad)
        if n_bad == len(is_bad):
            msg = "All data are low quality"
            raise ValidTimeStampError(msg)
        if n_bad:
            percentage = round(100 * n_bad / len(is_bad))
            logging.info(
                "Screening %s %% (%s/%s) data points with low quality",
                percentage,
                n_bad,
                len(is_bad),
            )
        self.data[self.variable][is_bad] = ma.masked

    def _remove_duplicate_timestamps(self) -> None:
        _, ind = np.unique(self.data["time"], return_index=True)
        self.data = self.data[ind]

    def _read_header(self, file: BinaryIO) -> None:
        raise NotImplementedError

    def _read_data(self, file: BinaryIO) -> None:
        raise NotImplementedError

    def _add_zenith_angle(self) -> None:
        ele, _azi = _decode_angles(self.data["_instrument_angles"], self.version)
        self.data = rfn.append_fields(self.data, "zenith_angle", 90 - ele)


class HatproBinLwp(HatproBin):
    """HATPRO *.LWP (Liquid Water Path) binary file reader."""

    variable = "lwp"

    def _read_header(self, file) -> None:
        self.header = _read_from_file(
            file,
            [
                ("file_code", "<i4"),
                ("_n_samples", "<i4"),
                ("_lwp_min", "<f"),
                ("_lwp_max", "<f"),
                ("_time_reference", "<i4"),
                ("retrieval_method", "<i4"),
            ],
        )
        if self.header["file_code"] == 934501978:
            self.version = 1
        elif self.header["file_code"] == 934501000:
            self.version = 2
        else:
            msg = f'Unknown HATPRO version. {self.header["file_code"]}'
            raise ValueError(msg)

    def _read_data(self, file) -> None:
        self.data = _read_from_file(
            file,
            [
                ("time", "<i4"),
                ("_quality_flag", "b"),
                ("lwp", "<f"),
                ("_instrument_angles", "<f" if self.version == 1 else "<i4"),
            ],
            self.header["_n_samples"],
        )
        self.data["lwp"] *= G_TO_KG


class HatproBinIwv(HatproBin):
    """HATPRO *.IWV (Integrated Water Vapour) binary file reader."""

    variable = "iwv"

    def _read_header(self, file) -> None:
        self.header = _read_from_file(
            file,
            [
                ("file_code", "<i4"),
                ("_n_samples", "<i4"),
                ("_iwv_min", "<f"),
                ("_iwv_max", "<f"),
                ("_time_reference", "<i4"),
                ("retrieval_method", "<i4"),
            ],
        )
        if self.header["file_code"] == 594811068:
            self.version = 1
        elif self.header["file_code"] == 594811000:
            self.version = 2
        else:
            msg = f'Unknown HATPRO version. {self.header["file_code"]}'
            raise ValueError(msg)

    def _read_data(self, file) -> None:
        self.data = _read_from_file(
            file,
            [
                ("time", "<i4"),
                ("_quality_flag", "b"),
                ("iwv", "<f"),
                ("_instrument_angles", "<f" if self.version == 1 else "<i4"),
            ],
            self.header["_n_samples"],
        )


class HatproBinCombined:
    """Combine HATPRO objects that share values of the given dimensions."""

    header: dict[str, np.ndarray]
    data: dict[str, np.ndarray]

    def __init__(self, files: list[HatproBin]):
        self.header = {}
        if len(files) == 1:
            arr = files[0].data
        elif len(files) == 2:
            f1, f2 = files
            arr = rfn.join_by("time", f1.data, f2.data, "outer")
            arr = rfn.append_fields(
                arr,
                "zenith_angle",
                _combine_values(arr["zenith_angle1"], arr["zenith_angle2"]),
            )
            # Workaround because rfn.drop_fields seems to incorrectly drop mask...
            arr = rfn.rename_fields(
                arr,
                {"zenith_angle1": "_tmp1", "zenith_angle2": "_tmp2"},
            )
        else:
            msg = "Only implemented up to 2 files"
            raise NotImplementedError(msg)
        self.data = {field: arr[field] for field in arr.dtype.fields}


def _combine_values(arr1: ma.MaskedArray, arr2: ma.MaskedArray) -> ma.MaskedArray:
    if not ma.allequal(arr1, arr2):
        msg = "Inconsistent values"
        raise ValueError(msg)
    return ma.where(~arr1.mask, arr1, arr2)
