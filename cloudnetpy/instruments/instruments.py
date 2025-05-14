from dataclasses import dataclass


@dataclass
class Instrument:
    domain: str
    category: str
    model: str | None = None
    manufacturer: str | None = None
    wavelength: float | None = None
    frequency: float | None = None


POLLYXT = Instrument(
    manufacturer="TROPOS",
    domain="lidar",
    category="Raman lidar",
    model="PollyXT",
    wavelength=1064.0,
)

CL51 = Instrument(
    manufacturer="Vaisala",
    domain="lidar",
    category="ceilometer",
    model="CL51",
    wavelength=910.0,
)

CL31 = Instrument(
    manufacturer="Vaisala",
    domain="lidar",
    category="ceilometer",
    model="CL31",
    wavelength=910.0,
)

CS135 = Instrument(
    manufacturer="Campbell Scientific",
    domain="lidar",
    category="ceilometer",
    model="CS135",
    wavelength=905.0,
)

CT25K = Instrument(
    manufacturer="Vaisala",
    domain="lidar",
    category="ceilometer",
    model="CT25k",
    wavelength=905.0,
)

CL61D = Instrument(
    manufacturer="Vaisala",
    domain="lidar",
    category="ceilometer",
    model="CL61d",
    wavelength=910.55,
)

CHM15K = Instrument(
    manufacturer="Lufft",
    domain="lidar",
    category="ceilometer",
    model="CHM15k",
    wavelength=1064.0,
)

CHM15KX = Instrument(
    manufacturer="Lufft",
    domain="lidar",
    category="ceilometer",
    model="CHM15kx",
    wavelength=1064.0,
)

MIRA10 = Instrument(
    manufacturer="METEK",
    domain="radar",
    category="cloud radar",
    model="MIRA-10",
    frequency=9.4,  # 9.2 - 9.6 GHz
)

MIRA35 = Instrument(
    manufacturer="METEK",
    domain="radar",
    category="cloud radar",
    model="MIRA-35",
    frequency=35.5,
)

COPERNICUS = Instrument(
    manufacturer="RAL Space",
    domain="radar",
    category="cloud radar",
    model="Copernicus",
    frequency=34.960,
)

GALILEO = Instrument(
    manufacturer="RAL Space",
    domain="radar",
    category="cloud radar",
    model="Galileo",
    frequency=94.0,
)

FMCW94 = Instrument(
    manufacturer="RPG-Radiometer Physics",
    domain="radar",
    category="cloud radar",
    model="RPG-FMCW-94",
    frequency=94.0,
)

FMCW35 = Instrument(
    manufacturer="RPG-Radiometer Physics",
    domain="radar",
    category="cloud radar",
    model="RPG-FMCW-35",
    frequency=35.0,
)

BASTA = Instrument(
    domain="radar",
    category="cloud radar",
    model="BASTA",
    frequency=95.0,
)

HATPRO = Instrument(
    manufacturer="RPG-Radiometer Physics",
    domain="mwr",
    category="microwave radiometer",
    model="HATPRO",
)

LHATPRO = Instrument(
    manufacturer="RPG-Radiometer Physics",
    domain="mwr",
    category="microwave radiometer",
    model="LHATPRO",
)

LHUMPRO_U90 = Instrument(
    manufacturer="RPG-Radiometer Physics",
    domain="mwr",
    category="microwave radiometer",
    model="LHUMPRO-U90",
)

RADIOMETRICS = Instrument(
    manufacturer="Radiometrics",
    domain="mwr",
    category="microwave radiometer",
)

HALO = Instrument(
    manufacturer="HALO Photonics",
    domain="lidar",
    category="Doppler lidar",
    model="StreamLine",
)

WINDCUBE_WLS100S = Instrument(
    manufacturer="Vaisala",
    domain="lidar",
    category="Doppler lidar",
    model="WindCube WLS100S",
)

WINDCUBE_WLS200S = Instrument(
    manufacturer="Vaisala",
    domain="lidar",
    category="Doppler lidar",
    model="WindCube WLS200S",
)

WINDCUBE_WLS400S = Instrument(
    manufacturer="Vaisala",
    domain="lidar",
    category="Doppler lidar",
    model="WindCube WLS400S",
)

WINDCUBE_WLS70 = Instrument(
    manufacturer="Vaisala",
    domain="lidar",
    category="Doppler lidar",
    model="WindCube WLS70",
)

PARSIVEL2 = Instrument(
    manufacturer="OTT HydroMet",
    domain="disdrometer",
    category="disdrometer",
    model="Parsivel2",
)

THIES = Instrument(
    manufacturer="Thies Clima",
    domain="disdrometer",
    category="disdrometer",
    model="LNM",
)

PLUVIO2 = Instrument(
    manufacturer="OTT HydroMet",
    domain="rain-gauge",
    category="rain-gauge",
    model="Pluvio2",
)

PLUVIO2S = Instrument(
    manufacturer="OTT HydroMet",
    domain="rain-gauge",
    category="rain-gauge",
    model="Pluvio2S",
)

THIES_PT = Instrument(
    manufacturer="Thies Clima",
    domain="rain-gauge",
    category="rain-gauge",
    model="Precipitation Transmitter",
)

RAIN_E_H3 = Instrument(
    manufacturer="LAMBRECHT meteo GmbH",
    domain="rain-gauge",
    category="rain-gauge",
    model="rain[e]H3",
)

GENERIC_WEATHER_STATION = Instrument(
    domain="weather-station",
    category="weather station",
)

MRR_PRO = Instrument(
    manufacturer="METEK",
    domain="rain-radar",
    category="rain radar",
    model="MRR-PRO",
    frequency=24.23,
)

FD12P = Instrument(
    manufacturer="Vaisala",
    domain="weather-station",
    category="present weather sensor",
    model="FD12P",
)
