from dataclasses import dataclass


@dataclass
class Instrument:
    manufacturer: str = None
    domain: str = None
    category: str = None
    model: str = None
    wavelength: float = None
    frequency: float = None


POLLYXT = Instrument(
    manufacturer='TROPOS',
    domain='lidar',
    category='Raman lidar',
    model='PollyXT',
    wavelength=1064.0)

CL51 = Instrument(
    manufacturer='Vaisala',
    domain='lidar',
    category='ceilometer',
    model='CL51',
    wavelength=910.0)

CL31 = Instrument(
    manufacturer='Vaisala',
    domain='lidar',
    category='ceilometer',
    model='CL31',
    wavelength=910.0)

CT25K = Instrument(
    manufacturer='Vaisala',
    domain='lidar',
    category='ceilometer',
    model='CT25k',
    wavelength=905.0)

CL61D = Instrument(
    manufacturer='Vaisala',
    domain='lidar',
    category='ceilometer',
    model='CL61d',
    wavelength=910.55)

CHM15K = Instrument(
    manufacturer='Lufft',
    domain='lidar',
    category='ceilometer',
    model='CHM15k',
    wavelength=1064.0)

MIRA35 = Instrument(
    manufacturer='METEK',
    domain='radar',
    category='cloud radar',
    model='MIRA-35',
    frequency=35.5)

FMCW94 = Instrument(
    manufacturer='RPG-Radiometer Physics',
    domain='radar',
    category='cloud radar',
    model='RPG-FMCW-94',
    frequency=94.0
)

FMCW35 = Instrument(
    manufacturer='RPG-Radiometer Physics',
    domain='radar',
    category='cloud radar',
    model='RPG-FMCW-35',
    frequency=35.0
)

BASTA = Instrument(
    domain='radar',
    category='cloud radar',
    model='BASTA',
    frequency=95.0
)

HATPRO = Instrument(
    manufacturer='RPG-Radiometer Physics',
    domain='mwr',
    category='microwave radiometer',
    model='HATPRO',
)
