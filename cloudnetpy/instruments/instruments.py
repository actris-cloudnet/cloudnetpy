from dataclasses import dataclass
from typing import Optional


@dataclass
class Instrument:
    domain: str
    category: str
    model: str
    manufacturer: Optional[str] = None
    wavelength: Optional[float] = None
    frequency: Optional[float] = None


POLLYXT = Instrument(
    manufacturer="TROPOS",
    domain="lidar",
    category="Raman lidar",
    model="PollyXT",
    wavelength=1064.0,
)

CL51 = Instrument(
    manufacturer="Vaisala", domain="lidar", category="ceilometer", model="CL51", wavelength=910.0
)

CL31 = Instrument(
    manufacturer="Vaisala", domain="lidar", category="ceilometer", model="CL31", wavelength=910.0
)

CT25K = Instrument(
    manufacturer="Vaisala", domain="lidar", category="ceilometer", model="CT25k", wavelength=905.0
)

CL61D = Instrument(
    manufacturer="Vaisala", domain="lidar", category="ceilometer", model="CL61d", wavelength=910.55
)

CHM15K = Instrument(
    manufacturer="Lufft", domain="lidar", category="ceilometer", model="CHM15k", wavelength=1064.0
)

CHM15KX = Instrument(
    manufacturer="Lufft", domain="lidar", category="ceilometer", model="CHM15kx", wavelength=1064.0
)

MIRA35 = Instrument(
    manufacturer="METEK", domain="radar", category="cloud radar", model="MIRA-35", frequency=35.5
)

COPERNICUS = Instrument(
    manufacturer="RAL Space",
    domain="radar",
    category="cloud radar",
    model="Copernicus",
    frequency=34.960,
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

BASTA = Instrument(domain="radar", category="cloud radar", model="BASTA", frequency=95.0)

HATPRO = Instrument(
    manufacturer="RPG-Radiometer Physics",
    domain="mwr",
    category="microwave radiometer",
    model="HATPRO",
)

RADIOMETRICS = Instrument(
    manufacturer="Radiometrics",
    domain="mwr",
    category="microwave radiometer",
    model="MP3014",
)

HALO = Instrument(
    manufacturer="HALO Photonics",
    domain="lidar",
    category="Doppler lidar",
    model="StreamLine",
)
