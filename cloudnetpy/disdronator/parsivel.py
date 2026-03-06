import datetime
import re
from os import PathLike
from typing import TypeAlias

import cftime
import netCDF4
import numpy as np

ParsivelOutput: TypeAlias = tuple[list, dict[int, list]]

# pyAtmosLogger headers
PYATLO_HEADER = {
    "datetime_utc": "%Y-%m-%d %H:%M:%S",
    "rain rate [mm/h]": 1,
    "rain accum [mm]": 2,
    "wawa": 3,  # missing in some files
    "Z [dBz]": 7,
    "MOR visibility [m]": 8,
    "sample interval [s]": 9,
    "Signal amplitude": 10,
    "Number of detected particles": 11,
    "Temperature sensor [°C]": 12,
    "Serial number": 13,
    "IOP firmware version": 14,
    "Current heating system [A]": 16,
    "Power supply voltage in the sensor [V]": 17,
    "Sensor status": 18,
    "Station name": 22,
    "Rain amount absolute [mm]": 24,
    "Error code": 25,
    "N": 90,
    "v": 91,
    "M": 93,
    # Custom headers (Kenttärova and Vehmasmäki):
    "wawa [ww]": 4,
    "wawa [METAR]": 5,
    "wawa [NWS]": 6,
    "DSP firmware version": 15,
    "Start of measurement [DD.MM.YY_HH:MM:SS]": 19,
    "Sensor time [HH:MM:SS]": 20,
    "Sensor date [DD.MM.YY]": 21,
    "Station number": 23,
    "Temperature PCB [°C]": 26,
    "Temperature right sensor head [°C]": 27,
    "Temperature left sensor head [°C]": 28,
    "Rain intensity 16 bit low [mm/h]": 30,
    "Rain intensity 16 bit high [mm/h]": 31,
    "Rain accumulated 16 bit [mm]": 32,
    "Reflectivity 16 bit [dBZ]": 33,
    "Kinetic energy [J m-2 h-1)]": 34,
    "Snow depth intensity (vol equiv.) [mm/h]": 35,
    "Number of particles": 60,
    "Particle list (empty, see particle file)": None,
}

# Headers used in OTT's ASDO software.
# https://www.otthydromet.com/en/p-asdo-application-software-ott-parsivel/6610001432
ASDO_HEADER = {
    "Date": "%Y/%m/%d",
    "Time": "%H:%M:%S",
    "Intensity of precipitation (mm/h)": 1,
    "Precipitation since start (mm)": 2,
    "Radar reflectivity (dBz)": 7,
    "MOR Visibility (m)": 8,
    "Signal amplitude of Laserband": 10,
    "Number of detected particles": 11,
    "Temperature in sensor (°C)": 12,
    "Heating current (A)": 16,
    "Sensor voltage (V)": 17,
    "Kinetic Energy": 34,
    "Snow intensity (mm/h)": 35,
    "Weather code SYNOP WaWa": 3,
    "Weather code METAR/SPECI": 5,
    "Weather code NWS": 6,
    "Optics status": 18,
    "Spectrum": 93,
}


GRANADA_HEADERS = {
    "TIMESTAMP": '"%Y-%m-%d %H:%M:%S"',
    "RECORD": None,
    "rain_intensity": 1,
    "snow_intensity": 35,
    "precipitation": 24,
    "weather_code_wawa": 3,
    "radar_reflectivity": 7,
    "mor_visibility": 8,
    "kinetic_energy": 34,
    "signal_amplitude": 10,
    "sensor_temperature": 12,
    "pbc_temperature": 26,
    "right_temperature": 27,
    "left_temperature": 28,
    "heating_current": 16,
    "sensor_voltage": 17,
    "sensor_status": 18,
    "error_code": 25,
    "number_particles": 11,
    "N": 90,
    "V": 91,
    "spectrum": 93,
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
    # sensor_time(time): 20 ???
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
INT_KEYS = {3, 4, 8, 9, 10, 11, 12, 13, 18, 25, 26, 27, 28, 60}


def _read_lines(
    telegram: list[str | int],
    content: str,
    field_separator: str,
    decimal_separator: str,
) -> ParsivelOutput:
    # Expand spectra in ASDO files.
    content = re.sub(r"<SPECTRUM>([^>]*)</SPECTRUM>", _expand_spectrum, content)
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
            for t in telegram:
                if t in FLOAT_KEYS:
                    data[t].append(float(values[0].replace(decimal_separator, ".")))
                    values = values[1:]
                elif t in INT_KEYS:
                    data[t].append(int(values[0]))
                    values = values[1:]
                elif t in (90, 91):
                    data[t].append([float(x) for x in values[:32]])
                    values = values[32:]
                elif t == 93:
                    spectrum = [float(x) for x in values[:1024]]
                    data[t].append(np.reshape(spectrum, (32, 32)))
                    values = values[1024:]
                elif isinstance(t, str):
                    dt = datetime.datetime.strptime(values[0], t)
                    if "%H" in t and "%Y" not in t:
                        times.append(dt.time())
                    elif "%H" not in t:
                        dates.append(dt.date())
                    else:
                        datetimes.append(dt)
                    values = values[1:]
                elif t is None:
                    values = values[1:]
                else:
                    data[t].append(values[0])
                    values = values[1:]
        except ValueError:
            # TODO: handle different length arrays
            continue
    if not datetimes:
        datetimes = [
            datetime.datetime.combine(date, time)
            for date, time in zip(dates, times, strict=True)
        ]
    return datetimes, data


def _expand_spectrum(m: re.Match) -> str:
    if m[1] == "ZERO":
        return "0;" * 1024
    return ";".join([x if x else "0" for x in m[1].split(";")])


def _read_typ_op4a(msg: str) -> dict:
    lines = msg.splitlines()
    if lines[0] != "TYP OP4A":
        msg = "Invalid message"
        raise ValueError(msg)
    data: dict = {}
    for line in lines[1:]:
        key, value = line.split(":", maxsplit=1)
        num = int(key)
        if num in INT_KEYS:
            data[num] = int(value)
        elif num in FLOAT_KEYS:
            data[num] = float(value)
        elif num in (90, 91):
            data[num] = [float(x) for x in value.rstrip(";").split(";")]
        elif num == 93:
            spectrum = [int(x) for x in value.rstrip(";").split(";")]
            data[num] = np.reshape(spectrum, (32, 32))
    return data


def _read_pyatmoslogger(filename: str | PathLike) -> ParsivelOutput:
    with open(filename, errors="replace") as f:
        header = f.readline().rstrip("\r\n")
        content = f.read()
    v_header = ";".join(f"v{i:02}" for i in range(32))
    N_header = ";".join(f"N{i:02}" for i in range(32))
    M_header = ";".join(f"M_{i}_{j}" for i in range(32) for j in range(32))
    header = header.replace(N_header, "N").replace(v_header, "v").replace(M_header, "M")
    header = header.replace("[�C]", "[°C]")  # some files latin1, some utf8
    telegram = [PYATLO_HEADER[key] for key in header.split(";")]
    return _read_lines(telegram, content, ";", ".")


def _read_asdo(filename: str | PathLike) -> ParsivelOutput:
    with open(filename, encoding="latin1") as f:
        headers = f.readline().rstrip("\r\n").split(";")
        content = f.read()
    telegram = [ASDO_HEADER[header] for header in headers]
    return _read_lines(telegram, content, ";", ",")


def _read_granada(filename: str | PathLike) -> ParsivelOutput:
    with open(filename) as f:
        _, header, _, _ = (
            f.readline(),
            f.readline().rstrip("\r\n"),
            f.readline(),
            f.readline(),
        )
        content = f.read()
    v_header = ",".join(f'"V({i + 1})"' for i in range(32))
    N_header = ",".join(f'"N({i + 1})"' for i in range(32))
    M_header = ",".join(f'"spectrum({i + 1})"' for i in range(1024))
    header = (
        header.replace(N_header, "N")
        .replace(v_header, "V")
        .replace(M_header, "spectrum")
    )
    telegram = [GRANADA_HEADERS[key.strip('"')] for key in header.split(",")]
    return _read_lines(telegram, content, ",", ".")


def _read_headerless(
    filename: str | PathLike,
    telegram: list[int | str],
    field_separator: str,
    decimal_separator: str,
) -> ParsivelOutput:
    with open(filename) as f:
        return _read_lines(telegram, f.read(), field_separator, decimal_separator)


def _read_hyytiala(filename: str | PathLike) -> ParsivelOutput:
    time: list = []
    data: dict = {}
    with open(filename) as f:
        content = f.read()
    for m in re.finditer(
        r"\[(?P<year>\d+)-(?P<month>\d+)-(?P<day>\d+) "
        r"(?P<hour>\d+):(?P<minute>\d+):(?P<second>\d+)\r?\n"
        r"(?P<output>[^\]]*)\]",
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
        # The data logger attempts to convert temperature from K to °C, but this
        # is incorrectly done only for the first value.
        data[12][0] -= 273
        # Sensor serial number from global attribute.
        data[13] = np.repeat(nc.Sensor_ID, len(time))
        return time, data


def _read_disdrodl(filename: str | PathLike) -> ParsivelOutput:
    with netCDF4.Dataset(filename) as nc:
        time = cftime.num2pydate(nc["time"][:], units=nc["time"].units)
        data = {
            num: nc[key][:] for key, num in DISDRODL_KEYS.items() if key in nc.variables
        }
        data[9] = np.repeat(nc["time_interval"][:], len(time))
        data[13] = np.repeat(nc.sensor_serial_number, len(time))
        return time, data


def _read_munich(filename: str | PathLike) -> ParsivelOutput:
    with netCDF4.Dataset(filename) as nc:
        time = cftime.num2pydate(nc["time"][:], units=nc["time"].units)
        data = {num: nc[key][:] for key, num in MUNICH_KEYS.items()}
        data[93] = np.transpose(data[93], (0, 2, 1))
        return time, data


def _read_leipzig(filename: str | PathLike) -> ParsivelOutput:
    with netCDF4.Dataset(filename) as nc:
        time = cftime.num2pydate(nc["Meas_Time"][:], units="seconds since 1970-01-01")
        data = {num: nc[key][:] for key, num in LEIPZIG_KEYS.items()}
        data[13] = np.repeat(nc.Sensor_ID, len(time))
        data[22] = np.repeat(nc.Station_Name, len(time))
        data[23] = np.repeat(nc.Station_ID, len(time))
        return time, data


def read_parsivel(
    filename: str | PathLike,
    telegram: list[int | str] | None = None,
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
    return _read_headerless(filename, telegram, field_separator, decimal_separator)


# Formats:
# raw telegram
# pyatmoslogger
# palaiseau/potenza asdo with header
# warsaw asdo without header?
# hyytiälä ascii
# granada toa5
# parsivel2nc: https://github.com/lacros-tropos/parsivel2tools
# disdroDL: https://github.com/ruisdael-observatory/disdroDL
# munich nc
# old leipzig nc

site_calib = {
    "ny-alesund": {
        "telegram": [
            "%Y%m%d%H%M%S.%f",
            1,
            2,
            3,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            16,
            17,
            18,
            22,
            24,
            25,
            90,
            91,
            93,
        ],
    },
    "warsaw": {
        "telegram": [
            "%d.%m.%Y",
            "%H:%M:%S",
            1,
            2,
            3,
            5,
            6,
            7,
            8,
            10,
            11,
            12,
            16,
            17,
            34,
            18,
            93,
        ],
        "decimal_separator": ",",
    },
}

# fmt: off
# mm
Dmid = np.array([
    0.062, 0.187, 0.312, 0.437, 0.562, 0.687, 0.812, 0.937, 1.062, 1.187, 1.375,
    1.625, 1.875, 2.125, 2.375, 2.750, 3.250, 3.750, 4.250, 4.750, 5.500, 6.500,
    7.500, 8.500, 9.500, 11.000, 13.000, 15.000, 17.000, 19.000, 21.500, 24.500,
])
# mm
Dspr = np.array( [
    0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.250,
    0.250, 0.250, 0.250, 0.250, 0.500, 0.500, 0.500, 0.500, 0.500, 1.000, 1.000,
    1.000, 1.000, 1.000, 2.000, 2.000, 2.000, 2.000, 2.000, 3.000, 3.000,
])
# m/s
Vmid = np.array([
    0.050, 0.150, 0.250, 0.350, 0.450, 0.550, 0.650, 0.750, 0.850, 0.950, 1.100,
    1.300, 1.500, 1.700, 1.900, 2.200, 2.600, 3.000, 3.400, 3.800, 4.400, 5.200,
    6.000, 6.800, 7.600, 8.800, 10.400, 12.000, 13.600, 15.200, 17.600, 20.800,
])
# m/s
Vspr = np.array([
    0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.100, 0.200,
    0.200, 0.200, 0.200, 0.200, 0.400, 0.400, 0.400, 0.400, 0.400, 0.800, 0.800,
    0.800, 0.800, 0.800, 1.600, 1.600, 1.600, 1.600, 1.600, 3.200, 3.200,
])
# fmt: on

A = 0.0054  # sampling area [m2] 30 mm * 180 mm
