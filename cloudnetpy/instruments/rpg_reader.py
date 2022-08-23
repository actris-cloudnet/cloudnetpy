from collections import namedtuple
from typing import BinaryIO, Dict, List, Literal, Tuple

import numpy as np
from numpy import ma
from numpy.lib import recfunctions as rfn


class Fmcw94Bin:
    """RPG Cloud Radar Level 1 data reader."""

    def __init__(self, filename):
        self.filename = filename
        self._file_position = 0
        self.header = self.read_rpg_header()
        self.data = self.read_rpg_data()

    def read_rpg_header(self):
        """Reads the header or rpg binary file."""

        def append(names, dtype=np.int32, n_values=1):
            """Updates header dictionary."""
            for name in names:
                header[name] = np.fromfile(file, dtype, int(n_values))

        header = {}
        with open(self.filename, "rb") as file:
            append(("file_code", "_header_length"), np.int32)
            append(("_start_time", "_stop_time"), np.uint32)
            append(("program_number",))
            append(("model_number",))  # 0 = single polarization, 1 = dual pol.
            header["_program_name"] = self.read_string(file)
            header["_customer_name"] = self.read_string(file)
            append(
                (
                    "radar_frequency",
                    "antenna_separation",
                    "antenna_diameter",
                    "antenna_gain",  # linear
                    "half_power_beam_width",
                ),
                np.float32,
            )
            append(("dual_polarization",), np.int8)  # 0 = single pol, 1 = LDR, 2 = STSR
            append(("sample_duration",), np.float32)
            append(("latitude", "longitude"), np.float32)
            append(
                (
                    "calibration_interval",
                    "_number_of_range_gates",
                    "_number_of_temperature_levels",
                    "_number_of_humidity_levels",
                    "_number_of_chirp_sequences",
                )
            )
            append(("range",), np.float32, int(header["_number_of_range_gates"]))
            append(
                ("_temperature_levels",), np.float32, int(header["_number_of_temperature_levels"])
            )
            append(("_humidity_levels",), np.float32, int(header["_number_of_humidity_levels"]))
            append(
                ("number_of_spectral_samples", "chirp_start_indices", "number_of_averaged_chirps"),
                n_values=int(header["_number_of_chirp_sequences"]),
            )
            append(
                ("integration_time", "range_resolution", "nyquist_velocity"),
                np.float32,
                int(header["_number_of_chirp_sequences"]),
            )
            append(
                (
                    "_is_power_levelling",
                    "_is_spike_filter",
                    "_is_phase_correction",
                    "_is_relative_power_correction",
                ),
                np.int8,
            )
            append(
                ("FFT_window",), np.int8
            )  # 0=square, 1=parzen, 2=blackman, 3=welch, 4=slepian2, 5=slepian3
            append(("input_voltage_range",))
            append(("noise_threshold",), np.float32)
            # Fix for Level 1 version 4 files:
            if int(header["file_code"]) >= 889348:
                _ = np.fromfile(file, np.int32, 25)
                _ = np.fromfile(file, np.uint32, 10000)
            self._file_position = file.tell()
        return header

    @staticmethod
    def read_string(file_id):
        """Read characters from binary data until whitespace."""
        str_out = ""
        while True:
            c = np.fromfile(file_id, np.int8, 1)
            if len(c) == 1 and c[0] < 0:
                c = [63]
            if len(c) == 0 or c[0] == 0:
                break
            str_out += chr(c[0])
        return str_out

    def read_rpg_data(self):
        """Reads the actual data from rpg binary file."""
        Dimensions = namedtuple("Dimensions", ["n_samples", "n_gates", "n_layers_t", "n_layers_h"])

        def _create_dimensions():
            """Returns possible lengths of the data arrays."""
            n_samples = np.fromfile(file, np.int32, 1)
            return Dimensions(
                int(n_samples),
                int(self.header["_number_of_range_gates"]),
                int(self.header["_number_of_temperature_levels"]),
                int(self.header["_number_of_humidity_levels"]),
            )

        def _create_variables():
            """Initializes dictionaries for data arrays."""
            vrs = {
                "_sample_length": np.zeros(dims.n_samples, int),
                "time": np.zeros(dims.n_samples, int),
                "time_ms": np.zeros(dims.n_samples, int),
                "quality_flag": np.zeros(dims.n_samples, int),
            }

            block1_vars = dict.fromkeys(
                (
                    "rain_rate",
                    "relative_humidity",
                    "temperature",
                    "pressure",
                    "wind_speed",
                    "wind_direction",
                    "voltage",
                    "brightness_temperature",
                    "lwp",
                    "if_power",
                    "elevation",
                    "azimuth_angle",
                    "status_flag",
                    "transmitted_power",
                    "transmitter_temperature",
                    "receiver_temperature",
                    "pc_temperature",
                )
            )

            block2_vars = dict.fromkeys(("Zh", "v", "width", "skewness", "kurtosis"))

            if int(self.header["dual_polarization"][0]) == 1:
                block2_vars.update(dict.fromkeys(("ldr", "rho_cx", "phi_cx")))
            elif int(self.header["dual_polarization"][0]) == 2:
                block2_vars.update(
                    dict.fromkeys(
                        (
                            "zdr",
                            "rho_hv",
                            "phi_dp",
                            "_",
                            "sldr",
                            "srho_hv",
                            "kdp",
                            "differential_attenuation",
                        )
                    )
                )
            return vrs, block1_vars, block2_vars

        def _add_sensitivities():
            ind0 = len(block1) + n_dummy
            ind1 = ind0 + dims.n_gates
            block1["_sensitivity_limit_v"] = float_block1[:, ind0:ind1]
            if int(self.header["dual_polarization"][0]) > 0:
                block1["_sensitivity_limit_h"] = float_block1[:, ind1:]

        def _get_length_of_dummy_data():
            return 3 + dims.n_layers_t + 2 * dims.n_layers_h

        def _get_length_of_sensitivity_data():
            if int(self.header["dual_polarization"][0]) > 0:
                return 2 * dims.n_gates
            return dims.n_gates

        def _get_float_block_lengths():
            block_one_length = len(block1) + n_dummy + n_sens
            block_two_length = len(block2)
            return block_one_length, block_two_length

        def _init_float_blocks():
            block_one = np.zeros((dims.n_samples, n_floats1))
            block_two = np.zeros((dims.n_samples, dims.n_gates, n_floats2))
            return block_one, block_two

        with open(self.filename, "rb") as file:
            file.seek(self._file_position)
            dims = _create_dimensions()
            aux, block1, block2 = _create_variables()
            n_dummy = _get_length_of_dummy_data()
            n_sens = _get_length_of_sensitivity_data()
            n_floats1, n_floats2 = _get_float_block_lengths()
            float_block1, float_block2 = _init_float_blocks()

            for sample in range(dims.n_samples):
                aux["_sample_length"][sample] = np.fromfile(file, np.int32, 1)
                aux["time"][sample] = np.fromfile(file, np.uint32, 1)
                aux["time_ms"][sample] = np.fromfile(file, np.int32, 1)
                aux["quality_flag"][sample] = np.fromfile(file, np.int8, 1)
                float_block1[sample, :] = np.fromfile(file, np.float32, n_floats1)
                is_data = np.fromfile(file, np.int8, dims.n_gates)
                is_data_ind = np.where(is_data)[0]
                n_valid = len(is_data_ind)
                values = np.fromfile(file, np.float32, n_floats2 * n_valid)
                float_block2[sample, is_data_ind, :] = values.reshape(n_valid, n_floats2)
        for n, name in enumerate(block1):
            block1[name] = float_block1[:, n]
        _add_sensitivities()
        for n, name in enumerate(block2):
            block2[name] = float_block2[:, :, n]
        return {**aux, **block1, **block2}


def _read_from_file(
    file: BinaryIO, fields: List[Tuple[str, str]], count: int = 1
) -> ma.MaskedArray:
    arr = np.fromfile(file, np.dtype(fields), count)
    masked_arr = ma.array(arr)
    if count == 1:
        return masked_arr[0]
    return masked_arr


def _decode_angles(x: np.ndarray, version: Literal[1, 2]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decode elevation and azimuth angles.

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
        raise NotImplementedError(
            f"Known versions for angle encoding are 1 and 2, but received {version}"
        )

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
        self._add_zenith_angle()

    def screen_bad_profiles(self):
        is_bad = self.data["_quality_flag"] & 0b110 == self.QUALITY_LOW << 1
        self.data[self.variable][is_bad] = ma.masked

    def _read_header(self, file: BinaryIO):
        raise NotImplementedError()

    def _read_data(self, file: BinaryIO):
        raise NotImplementedError()

    def _add_zenith_angle(self):
        ele, _azi = _decode_angles(self.data["_instrument_angles"], self.version)
        self.data = rfn.append_fields(self.data, "zenith_angle", 90 - ele)


class HatproBinLwp(HatproBin):
    """HATPRO *.LWP (Liquid Water Path) binary file reader."""

    variable = "lwp"

    def _read_header(self, file):
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
            raise ValueError(f'Unknown HATPRO version. {self.header["file_code"]}')

    def _read_data(self, file):
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


class HatproBinIwv(HatproBin):
    """HATPRO *.IWV (Integrated Water Vapour) binary file reader."""

    variable = "iwv"

    def _read_header(self, file):
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
            raise ValueError(f'Unknown HATPRO version. {self.header["file_code"]}')

    def _read_data(self, file):
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

    header: Dict[str, np.ndarray]
    data: Dict[str, np.ndarray]

    def __init__(self, files: List[HatproBin]):
        self.header = {}
        if len(files) == 1:
            arr = files[0].data
        elif len(files) == 2:
            f1, f2 = files
            arr = rfn.join_by("time", f1.data, f2.data, "outer")
            arr = rfn.append_fields(
                arr, "zenith_angle", _combine_values(arr["zenith_angle1"], arr["zenith_angle2"])
            )
            # Workaround because rfn.drop_fields seems to incorrectly drop mask...
            # arr = rfn.drop_fields(arr, ["zenith_angle1", "zenith_angle2"])
            arr = rfn.rename_fields(arr, {"zenith_angle1": "_tmp1", "zenith_angle2": "_tmp2"})
        else:
            raise NotImplementedError("Only implemented up to 2 files")
        self.data = {field: arr[field] for field in arr.dtype.fields}


def _combine_values(arr1: ma.MaskedArray, arr2: ma.MaskedArray) -> ma.MaskedArray:
    if not ma.allequal(arr1, arr2):
        raise ValueError("Inconsistent values")
    return ma.where(~arr1.mask, arr1, arr2)
