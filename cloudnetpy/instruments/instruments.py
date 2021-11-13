
class Instrument:
    def __init__(self,
                 manufacturer: str = None,
                 model: str = None,
                 type: str = None,
                 domain: str = None,
                 wavelength: float = None,
                 frequency: float = None):
        self.manufacturer = manufacturer
        self.model = model
        self.type = type
        self.domain = domain
        self.wavelength = wavelength
        self.frequency = frequency


POLLYXT = Instrument(
    model='PollyXT',
    type='Raman lidar',
    domain='lidar',
    wavelength=1064.0)

CL51 = Instrument(
    manufacturer='Vaisala',
    model='CL51',
    type='ceilometer',
    domain='lidar',
    wavelength=910.0)

CL31 = Instrument(
    manufacturer='Vaisala',
    model='CL31',
    type='ceilometer',
    domain='lidar',
    wavelength=910.0)

CT25K = Instrument(
    manufacturer='Vaisala',
    model='CT25k',
    type='ceilometer',
    domain='lidar',
    wavelength=905.0)

CL61D = Instrument(
    manufacturer='Vaisala',
    model='CL61d',
    type='ceilometer',
    domain='lidar',
    wavelength=910.55)

CHM15K = Instrument(
    manufacturer='Lufft',
    model='CHM15k',
    type='ceilometer',
    domain='lidar',
    wavelength=1064.0)

MIRA35 = Instrument(
    manufacturer='METEK',
    model='MIRA-35',
    type='cloud radar',
    domain='radar',
    frequency=35.5)

FMCW94 = Instrument(
    manufacturer='RPG-Radiometer Physics',
    model='RPG-FMCW-94',
    type='cloud radar',
    domain='radar',
    frequency=94.0
)

HATPRO = Instrument(
    manufacturer='RPG-Radiometer Physics',
    model='HATPRO',
    type='microwave radiometer',
    domain='mwr'
)

