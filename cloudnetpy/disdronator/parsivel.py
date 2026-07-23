import datetime
import re
from collections.abc import Sequence
from os import PathLike
from typing import TypeAlias

import cftime
import netCDF4
import numpy as np
import numpy.typing as npt

from cloudnetpy.disdronator.process import DisdroL1
from cloudnetpy.disdronator.utils import convert_to_numpy

ParsivelOutput: TypeAlias = tuple[list, dict[int, list]]

# pyAtmosLogger headers
PYATLO_HEADER: dict[bytes, str | int | None] = {
    b"datetime_utc": "%Y-%m-%d %H:%M:%S",
    b"rain rate [mm/h]": 1,
    b"rain accum [mm]": 2,
    b"wawa": 3,  # missing in some files
    b"Z [dBz]": 7,
    b"MOR visibility [m]": 8,
    b"sample interval [s]": 9,
    b"Signal amplitude": 10,
    b"Number of detected particles": 11,
    b"Temperature sensor [\xc2\xb0C]": 12,  # utf8
    b"Temperature sensor [\xb0C]": 12,  # latin1
    b"Serial number": 13,
    b"IOP firmware version": 14,
    b"Current heating system [A]": 16,
    b"Power supply voltage in the sensor [V]": 17,
    b"Sensor status": 18,
    b"Station name": 22,
    b"Rain amount absolute [mm]": 24,
    b"Error code": 25,
    b"N": 90,
    b"v": 91,
    b"M": 93,
    # Custom headers (Kenttärova and Vehmasmäki):
    b"wawa [ww]": 4,
    b"wawa [METAR]": 5,
    b"wawa [NWS]": 6,
    b"DSP firmware version": 15,
    b"Start of measurement [DD.MM.YY_HH:MM:SS]": 19,
    b"Sensor time [HH:MM:SS]": 20,
    b"Sensor date [DD.MM.YY]": 21,
    b"Station number": 23,
    b"Temperature PCB [\xc2\xb0C]": 26,  # utf8
    b"Temperature PCB [\xb0C]": 26,  # latin1
    b"Temperature right sensor head [\xc2\xb0C]": 27,  # utf8
    b"Temperature right sensor head [\xb0C]": 27,  # latin1
    b"Temperature left sensor head [\xc2\xb0C]": 28,  # utf8
    b"Temperature left sensor head [\xb0C]": 28,  # utf8
    b"Rain intensity 16 bit low [mm/h]": 30,
    b"Rain intensity 16 bit high [mm/h]": 31,
    b"Rain accumulated 16 bit [mm]": 32,
    b"Reflectivity 16 bit [dBZ]": 33,
    b"Kinetic energy [J m-2 h-1)]": 34,
    b"Snow depth intensity (vol equiv.) [mm/h]": 35,
    b"Number of particles": 60,
    b"Particle list (empty, see particle file)": None,
}

# Headers used in OTT's ASDO software.
# https://www.otthydromet.com/en/p-asdo-application-software-ott-parsivel/6610001432
ASDO_HEADER: dict[bytes, str | int | None] = {
    b"Date": "%Y/%m/%d",
    b"Time": "%H:%M:%S",
    b"Intensity of precipitation (mm/h)": 1,
    b"Precipitation since start (mm)": 2,
    b"Radar reflectivity (dBz)": 7,
    b"MOR Visibility (m)": 8,
    b"Signal amplitude of Laserband": 10,
    b"Number of detected particles": 11,
    b"Temperature in sensor (\xc2\xb0C)": 12,  # utf8
    b"Temperature in sensor (\xb0C)": 12,  # latin1
    b"Heating current (A)": 16,
    b"Sensor voltage (V)": 17,
    b"Kinetic Energy": 34,
    b"Snow intensity (mm/h)": 35,
    b"Weather code SYNOP WaWa": 3,
    b"Weather code METAR/SPECI": 5,
    b"Weather code NWS": 6,
    b"Optics status": 18,
    b"Spectrum": 93,
}

# Headers used in Granada's TOA5 format from CR1000 data logger.
GRANADA_HEADERS: dict[bytes, str | int | None] = {
    b"TIMESTAMP": '"%Y-%m-%d %H:%M:%S"',
    b"RECORD": None,
    # New snake case names:
    b"rain_intensity": 1,
    b"snow_intensity": 35,
    b"precipitation": 24,
    b"weather_code_wawa": 3,
    b"radar_reflectivity": 7,
    b"mor_visibility": 8,
    b"kinetic_energy": 34,
    b"signal_amplitude": 10,
    b"sensor_temperature": 12,
    b"pbc_temperature": 26,  # sic
    b"right_temperature": 27,
    b"left_temperature": 28,
    b"heating_current": 16,
    b"sensor_voltage": 17,
    b"sensor_status": 18,
    b"error_code": 25,
    b"number_particles": 11,
    b"N": 90,
    b"V": 91,
    b"spectrum": 93,
    # Old camel case names:
    b"rainIntensity": 1,
    b"snowIntensity": 35,
    b"accPrec": 24,
    b"weatherCodeWaWa": 3,
    b"radarReflectivity": 7,
    b"morVisibility": 8,
    b"kineticEnergy": 34,
    b"signalAmplitude": 10,
    b"sensorTemperature": 12,
    b"pbcTemperature": 26,  # sic
    b"rightTemperature": 27,
    b"leftTemperature": 28,
    b"heatingCurrent": 16,
    b"sensorVoltage": 17,
    b"sensorStatus": 18,
    b"errorCode": 25,
    b"numberParticles": 11,
}

# parsivel2nc
# https://github.com/lacros-tropos/parsivel2tools
PARSIVEL2NC_KEYS = {
    "interval": 9,
    "data_raw": 93,
    "number_concentration": 90,
    "fall_velocity": 91,
    "n_particles": 11,
    "rainfall_rate": 1,
    "radar_reflectivity": 7,
    "E_kin": 24,
    "visibility": 8,
    "synop_WaWa": 3,
    "synop_WW": 4,
    "T_sensor": 12,
    "sig_laser": 10,
    "state_sensor": 18,
    "V_sensor": 17,
    "I_heating": 16,
    "error_code": 25,
}

# disdroDL
# https://github.com/ruisdael-observatory/disdroDL/blob/main/configs_netcdf/config_general_parsivel.yml
DISDRODL_KEYS = {
    "rain_intensity": 1,
    "code_4680": 3,
    "code_4677": 4,
    "code_4678": 5,
    "code_NWS": 6,
    "reflectivity": 7,
    "MOR": 8,
    "amplitude": 10,
    "n_particles": 11,
    "T_sensor": 12,
    "I_heating": 16,
    "V_power_supply": 17,
    "state_sensor": 18,
    "absolute_rain_amount": 24,
    "error_code": 25,
    "T_pcb": 26,
    "T_L_sensor_head": 27,
    "T_R_sensor_head": 28,
    "kinetic_energy": 34,
    "snowfall_intensity": 35,
    "fieldN": 90,
    "fieldV": 91,
    "data_raw": 93,
}

# Similar to netCDF files from pyAtmosLogger but with differences in variables
# names and without any global attributes.
MUNICH_KEYS = {
    "status_sensor": 18,
    "sensor_time": 20,
    "error_code": 25,
    "rr": 1,
    "rain_accum": 2,
    "wawa": 3,
    "Ze": 7,
    "n_particles": 11,
    "snow_intensity": 35,
    "sample_interval": 9,
    "serial_no": 13,
    "firmware_IOP": 14,
    "firmware_DSP": 15,
    "curr_heating": 16,
    "volt_sensor": 17,
    "signal_amplitude": 10,
    "T_sensor_housing": 12,
    "T_pcb": 26,
    "T_sensor_right": 27,
    "T_sensor_left": 28,
    "N": 90,
    "v": 91,
    "M": 93,
}

# Possibly an old version of parsivel2nc used in older Leipzig files.
LEIPZIG_KEYS = {
    "Meas_Interval": 9,
    "RR_Intensity": 1,
    "RR_Accumulated": 2,
    "RR_Total": 24,
    "Synop_WaWa": 3,
    "Synop_WW": 4,
    "Reflectivity": 7,
    "Visibility": 8,
    "T_Sensor": 12,
    "Sig_Laser": 10,
    "N_Particles": 11,
    "State_Sensor": 18,
    "E_kin": 24,
    "V_Sensor": 17,
    "I_Heating": 16,
    "Error_Code": 25,
    "Data_N_Field": 90,
    "Data_V_Field": 91,
    "Data_Raw": 93,
}

FLOAT_KEYS = {1, 2, 7, 16, 17, 24, 30, 31, 33, 34, 35}
INT_KEYS = {3, 4, 8, 9, 10, 11, 12, 18, 25, 26, 27, 28, 60}


def _read_lines(
    telegram: Sequence[str | int | None],
    content: bytes,
    field_separator: bytes,
    decimal_separator: bytes,
) -> ParsivelOutput:
    # Expand spectra in ASDO files.
    content = re.sub(rb"<SPECTRUM>([^>]*)</SPECTRUM>", _expand_spectrum, content)
    expected_len = 0
    for t in telegram:
        if t == 90 or t == 91:
            expected_len += 32
        elif t == 93:
            expected_len += 1024
        else:
            expected_len += 1
    data: dict = {t: [] for t in telegram if isinstance(t, int)}
    times = []
    dates = []
    datetimes = []
    for line in content.splitlines():
        values = line.rstrip(field_separator).split(field_separator)
        if len(values) != expected_len:
            continue
        try:
            row_time = None
            row_date = None
            row_datetime = None
            row_data: dict = {}
            for t in telegram:
                if t in FLOAT_KEYS:
                    row_data[t] = float(values[0].replace(decimal_separator, b"."))
                    values = values[1:]
                elif t in INT_KEYS:
                    row_data[t] = int(values[0])
                    values = values[1:]
                elif t in (90, 91):
                    row_data[t] = [float(x) for x in values[:32]]
                    values = values[32:]
                elif t == 93:
                    spectrum = [int(x) for x in values[:1024]]
                    row_data[t] = np.reshape(spectrum, (32, 32))
                    values = values[1024:]
                elif isinstance(t, str):
                    dt = datetime.datetime.strptime(values[0].decode(), t)
                    if "%H" in t and "%Y" not in t:
                        row_time = dt.time()
                    elif "%H" not in t:
                        row_date = dt.date()
                    else:
                        row_datetime = dt
                    values = values[1:]
                elif t is None:
                    values = values[1:]
                else:
                    row_data[t] = values[0]
                    values = values[1:]
            if row_time is not None:
                times.append(row_time)
            if row_date is not None:
                dates.append(row_date)
            if row_datetime is not None:
                datetimes.append(row_datetime)
            for t, value in row_data.items():
                data[t].append(value)
        except ValueError:
            continue
    if not datetimes:
        datetimes = [
            datetime.datetime.combine(date, time)
            for date, time in zip(dates, times, strict=True)
        ]
    return datetimes, data


def _expand_spectrum(m: re.Match) -> bytes:
    if m[1] == b"ZERO":
        return b"0;" * 1024
    return b";".join([x or b"0" for x in m[1].split(b";")[:1024]]) + b";"


def _read_typ_op4a(content: bytes) -> dict:
    lines = content.splitlines()
    if lines[0] != b"TYP OP4A":
        msg = "Invalid message"
        raise ValueError(msg)
    data: dict = {}
    for line in lines[1:]:
        key, value = line.split(b":", maxsplit=1)
        num = int(key)
        if num in INT_KEYS:
            data[num] = int(value)
        elif num in FLOAT_KEYS:
            data[num] = float(value)
        elif num in (90, 91):
            data[num] = [float(x) for x in value.rstrip(b";").split(b";")]
        elif num == 93:
            spectrum = [int(x) for x in value.rstrip(b";").split(b";")]
            data[num] = np.reshape(spectrum, (32, 32))
        else:
            data[num] = value
    return data


def _read_pyatmoslogger(filename: str | PathLike) -> ParsivelOutput:
    with open(filename, "rb") as f:
        header = f.readline().rstrip(b"\r\n")
        content = f.read()
    v_header = ";".join(f"v{i:02}" for i in range(32))
    N_header = ";".join(f"N{i:02}" for i in range(32))
    M_header = ";".join(f"M_{i}_{j}" for i in range(32) for j in range(32))
    header = (
        header.replace(N_header.encode(), b"N")
        .replace(v_header.encode(), b"v")
        .replace(M_header.encode(), b"M")
    )
    telegram = [PYATLO_HEADER[key] for key in header.split(b";")]
    return _read_lines(telegram, content, b";", b".")


def _read_asdo(filename: str | PathLike) -> ParsivelOutput:
    with open(filename, "rb") as f:
        headers = f.readline().rstrip(b"\r\n").split(b";")
        content = f.read()
    telegram = [ASDO_HEADER[header] for header in headers]
    return _read_lines(telegram, content, b";", b",")


def _read_granada(filename: str | PathLike) -> ParsivelOutput:
    with open(filename, "rb") as f:
        _, header, _, _ = (
            f.readline(),
            f.readline().rstrip(b"\r\n"),
            f.readline(),
            f.readline(),
        )
        content = f.read()
    v_header = ",".join(f'"V({i + 1})"' for i in range(32))
    N_header = ",".join(f'"N({i + 1})"' for i in range(32))
    M_header = ",".join(f'"spectrum({i + 1})"' for i in range(1024))
    header = (
        header.replace(N_header.encode(), b"N")
        .replace(v_header.encode(), b"V")
        .replace(M_header.encode(), b"spectrum")
    )
    telegram = [GRANADA_HEADERS[key.strip(b'"')] for key in header.split(b",")]
    return _read_lines(telegram, content, b",", b".")


def _read_headerless(
    filename: str | PathLike,
    telegram: Sequence[int | str | None],
    field_separator: bytes,
    decimal_separator: bytes,
) -> ParsivelOutput:
    with open(filename, "rb") as f:
        content = f.read()
    return _read_lines(telegram, content, field_separator, decimal_separator)


def _read_hyytiala(filename: str | PathLike) -> ParsivelOutput:
    time: list = []
    data: dict = {}
    with open(filename, "rb") as f:
        content = f.read()
    for m in re.finditer(
        rb"\[(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+) "
        rb"(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+)\r?\n"
        rb"(?P<output>[^\]]*)\]",
        content,
    ):
        try:
            record = _read_typ_op4a(m["output"])
            timestamp = datetime.datetime(
                int(m["year"]),
                int(m["month"]),
                int(m["day"]),
                int(m["hour"]),
                int(m["minute"]),
                int(m["second"]),
            )
        except ValueError:
            continue
        for key in record:
            if key not in data:
                data[key] = [None] * len(time)
        for key in data:
            data[key].append(record.get(key))
        time.append(timestamp)
    return time, data


def _read_parsivel2nc(filename: str | PathLike) -> ParsivelOutput:
    with netCDF4.Dataset(filename) as nc:
        time = cftime.num2pydate(nc["time"][:], units=nc["time"].units)
        data = {num: nc[key][:] for key, num in PARSIVEL2NC_KEYS.items()}
        # The data logger converts mm/h to m/s, so we need to revert this.
        data[1] *= 3600 * 1000
        # The data logger attempts to convert temperature from °C to K, but this
        # is incorrectly done only for the first value.
        data[12][0] -= 273
        # Sensor serial number from global attribute.
        data[13] = np.repeat(int(nc.Sensor_ID), len(time))
        # Convert raw data from double to integer.
        data[93] = data[93].astype(np.int32)
        return time, data


def _read_disdrodl(filename: str | PathLike) -> ParsivelOutput:
    with netCDF4.Dataset(filename) as nc:
        time = cftime.num2pydate(nc["time"][:], units=nc["time"].units)
        data = {
            num: nc[key][:] for key, num in DISDRODL_KEYS.items() if key in nc.variables
        }
        data[9] = np.repeat(nc["time_interval"][:], len(time))
        data[13] = np.repeat(int(nc.sensor_serial_number), len(time))
        return time, data


def _read_munich(filename: str | PathLike) -> ParsivelOutput:
    with netCDF4.Dataset(filename) as nc:
        time = cftime.num2pydate(nc["time"][:], units=nc["time"].units)
        data = {num: nc[key][:] for key, num in MUNICH_KEYS.items()}
        data[93] = np.swapaxes(data[93], 1, 2)
        return time, data


def _read_leipzig(filename: str | PathLike) -> ParsivelOutput:
    with netCDF4.Dataset(filename) as nc:
        time = cftime.num2pydate(nc["Meas_Time"][:], units="seconds since 1970-01-01")
        data = {num: nc[key][:] for key, num in LEIPZIG_KEYS.items()}
        data[13] = np.repeat(int(nc.Sensor_ID), len(time))
        data[22] = np.repeat(nc.Station_Name, len(time))
        data[23] = np.repeat(nc.Station_ID, len(time))
        return time, data


def _read_parsivel(
    filename: str | PathLike,
    telegram: Sequence[int | str | None] | None = None,
    field_separator: str = ";",
    decimal_separator: str = ".",
) -> ParsivelOutput:
    try:
        with netCDF4.Dataset(filename) as nc:
            if "number_concentration" in nc.variables:
                return _read_parsivel2nc(filename)
            if "fieldN" in nc.variables:
                return _read_disdrodl(filename)
            if "N" in nc.variables:
                return _read_munich(filename)
            if "Data_N_Field" in nc.variables:
                return _read_leipzig(filename)
            msg = "Unsupported netCDF file"
            raise ValueError(msg)
    except OSError:
        pass
    with open(filename, "rb") as f:
        head = f.read(50)
    if head.startswith(b"datetime_utc;"):
        return _read_pyatmoslogger(filename)
    if head.startswith(b"Date;Time;"):
        return _read_asdo(filename)
    if head.startswith(b'"TOA5"'):
        return _read_granada(filename)
    if b"TYP OP4A" in head:
        return _read_hyytiala(filename)
    if telegram is None:
        msg = "telegram must be defined"
        raise ValueError(msg)
    return _read_headerless(
        filename, telegram, field_separator.encode(), decimal_separator.encode()
    )


def read_parsivel(
    filename: str | PathLike,
    telegram: Sequence[int | str | None] | None = None,
    field_separator: str = ";",
    decimal_separator: str = ".",
) -> tuple[npt.NDArray, dict[int, npt.NDArray]]:
    time, data = _read_parsivel(filename, telegram, field_separator, decimal_separator)
    return np.array(time), convert_to_numpy(data, {}, INT_KEYS, FLOAT_KEYS)


def read_parsivel_l1(time: npt.NDArray, l0: dict[int, npt.NDArray]) -> DisdroL1:
    if 93 not in l0:
        msg = "No raw data"
        raise ValueError(msg)
    data_raw = np.swapaxes(l0[93], 1, 2)
    interval = (
        l0[9]
        if 9 in l0
        else np.median(np.diff(time.astype("datetime64[s]"))).astype(int)
    )
    return DisdroL1(
        diameter=D,
        diameter_bins=D_BINS,
        diameter_spread=D_SPREAD,
        velocity=V,
        velocity_bins=V_BINS,
        velocity_spread=V_SPREAD,
        time=time,
        interval=np.broadcast_to(interval, len(time)),
        area_nom=AREA_NOM,
        area_eff=AREA_EFF,
        data_raw=data_raw,
    )


# fmt: off
D = np.array([
    0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375, 1.0625,
    1.1875, 1.3750, 1.6250, 1.8750, 2.1250, 2.3750, 2.7500, 3.2500, 3.7500,
    4.2500, 4.7500, 5.5000, 6.5000, 7.5000, 8.5000, 9.5000, 11.0000, 13.0000,
    15.0000, 17.0000, 19.0000, 21.5000, 24.5000,
])
D_BINS = np.array([
    0.000, 0.125, 0.250, 0.375, 0.500, 0.625, 0.750, 0.875, 1.000, 1.125, 1.250,
    1.500, 1.750, 2.000, 2.250, 2.500, 3.000, 3.500, 4.000, 4.500, 5.000, 6.000,
    7.000, 8.000, 9.000, 10.000, 12.000, 14.000, 16.000, 18.000, 20.000, 23.000,
    26.000
])
D_SPREAD = np.array([
    0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.250,
    0.250, 0.250, 0.250, 0.250, 0.500, 0.500, 0.500, 0.500, 0.500, 1.000, 1.000,
    1.000, 1.000, 1.000, 2.000, 2.000, 2.000, 2.000, 2.000, 3.000, 3.000
])
V = np.array([
    0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.1, 1.30, 1.5,
    1.7, 1.9, 2.2, 2.6, 3., 3.4, 3.8, 4.4, 5.2, 6.00, 6.8, 7.6, 8.8, 10.4, 12.,
    13.6, 15.2, 17.6, 20.8
])
V_BINS = np.array([
    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8,
    2.0, 2.4, 2.8, 3.2, 3.6, 4.0, 4.8, 5.6, 6.4, 7.2, 8.0, 9.6, 11.2, 12.8,
    14.4, 16.0, 19.2, 22.4
])
V_SPREAD = np.array([
    0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2,
    0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.8, 0.8, 0.8, 1.6, 1.6, 1.6, 1.6, 1.6,
    3.2, 3.2
])
AREA_NOM = 180 * 30
AREA_EFF = 180 * (30 - D / 2)
# fmt: on
