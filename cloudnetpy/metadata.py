""" Metadata for Cloudnet variables. The values and definitions
in this module remain same for all Cloudnet files.
"""

from collections import namedtuple

_LOG = 'logarithmic'
_LIN = 'linear'

MetaData = namedtuple('MetaData', ['long_name',
                                   'units',
                                   'valid_range',
                                   'plot_range',
                                   'plot_scale',
                                   'comment'])

# Default values for namedtuple. Python 3.7 has better syntax for this.
MetaData.__new__.__defaults__ = (None,)*len(MetaData._fields)

ATTRIBUTES = {
    'time': MetaData(
        'Time UTC',
        'decimal hours since midnight'
    ),
    'model_time': MetaData(
        'model time UTC',
        'decimal hours since midnight'
    ),
    'height': MetaData(
        'Height above mean sea level',
        'm'
    ),
    'model_height': MetaData(
        'Height of model variables above mean sea level',
        'm'
    ),
    'range': MetaData(
        'Height above ground',
        'm'
    ),
    'latitude': MetaData(
        'Latitude of site',
        'degrees_north'
    ),
    'longitude': MetaData(
        'Longitude of site',
        'degrees_north'
    ),
    'altitude': MetaData(
        'Altitude of site',
        'm'
    ),
    'radar_frequency': MetaData(
        'Radar transmit frequency',
        'GHz'
    ),
    'ldr': MetaData(
        'Linear depolarisation ratio',
        'dB',
        (-35, 5),
        (-30, 0),
        _LIN
    ),
    'width': MetaData(
        'Spectral width',
        'm s-1',
        (0, 3),
        (0, 3),
        _LOG
    ),
    'v': MetaData(
        'Doppler velocity',
        'm s-1',
        (-10, 10),
        (-4, 2),
        _LIN
    ),
    'SNR': MetaData(
        'Signal-to-noise ratio',
        'dB',
        (-30, 70),
        (-20, 60),
        _LIN
    ),
    'Z': MetaData(
        'Radar reflectivity factor',
        'dBZ',
        (-60, 30),
        (-40, 20),
        _LIN
    ),

}

_DEFINITIONS = {
    'category_bits':
    ('\nBit 0: Small liquid droplets are present.\n'
     'Bit 1: Falling hydrometeors are present; if Bit 2 is set then these are most\n'
     '       likely ice particles, otherwise they are drizzle or rain drops.\n'
     'Bit 2: Wet-bulb temperature is less than 0 degrees C, implying\n'
     '       the phase of Bit-1 particles.\n'
     'Bit 3: Melting ice particles are present.\n'
     'Bit 4: Aerosol particles are present and visible to the lidar.\n'
     'Bit 5: Insects are present and visible to the radar.'),

    'quality_bits':
    ('\nBit 0: An echo is detected by the radar.\n'
     'Bit 1: An echo is detected by the lidar.\n'
     'Bit 2: The apparent echo detected by the radar is ground clutter\n'
     '       or some other non-atmospheric artifact.\n'
     'Bit 3: The lidar echo is due to clear-air molecular scattering.\n'
     'Bit 4: Liquid water cloud, rainfall or melting ice below this pixel\n'
     '       will have caused radar and lidar attenuation; if bit 5 is set then\n'
     '       a correction for the radar attenuation has been performed;\n'
     '       otherwise do not trust the absolute values of reflectivity factor.\n'
     '       No correction is performed for lidar attenuation.\n'
     'Bit 5: Radar reflectivity has been corrected for liquid-water attenuation\n'
     '       using the microwave radiometer measurements of liquid water path\n'
     '       and the lidar estimation of the location of liquid water cloud;\n'
     '       be aware that errors in reflectivity may result.')
}

_COMMENTS = {
    'category_bits':
    ('This variable contains information on the nature of the targets\n'
     'at each pixel, thereby facilitating the application of algorithms that work\n'
     'with only one type of target. The information is in the form of an array of\n'
     'bits, each of which states either whether a certain type of particle is present\n'
     '(e.g. aerosols), or the whether some of the target particles have a particular\n'
     'property. The definitions of each bit are given in the definition attribute.\n'
     'Bit 0 is the least significant.'),

    'quality_bits':
    ('This variable contains information on the quality of the\n'
     'data at each pixel. The information is in the form of an array\n'
     'of bits, and the definitions of each bit are given in the definition\n'
     'attribute. Bit 0 is the least significant'),

    'radar_liquid_atten':
    ('This variable was calculated from the liquid water path\n'
     'measured by microwave radiometer using lidar and radar returns to perform\n'
     'an approximate partioning of the liquid water content with height. Bit 5 of\n'
     'the quality_bits variable indicates where a correction for liquid water\n'
     'attenuation has been performed.'),

    'radar_gas_atten':
    ('This variable was calculated from the model temperature,\n'
     'pressure and humidity, but forcing pixels containing liquid cloud to saturation\n'
     'with respect to liquid water. It was calculated using the millimeter-wave propagation\n'
     'model of Liebe (1985, Radio Sci. 20(5), 1069-1089). It has been used to correct Z.'),

    'Tw':
    ('This variable was calculated from model T, P and relative humidity, which were first\n'
     'interpolated into measurement grid.'),

    'Z_sensitivity':
    ('This variable is an estimate of the radar sensitivity,\n'
     'i.e. the minimum detectable radar reflectivity, as a function\n'
     'of height. It includes the effect of ground clutter and gas attenuation\n'
     'but not liquid attenuation.'),

    'Z_error':
    ('This variable is an estimate of the one-standard-deviation\n'
     'random error in radar reflectivity factor. It originates\n'
     'from the following independent sources of error:\n'
     '1) Precision in reflectivity estimate due to finite signal to noise\n'
     '   and finite number of pulses\n'
     '2) 10% uncertainty in gaseous attenuation correction (mainly due to\n'
     '   error in model humidity field)\n'
     '3) Error in liquid water path (given by the variable lwp_error) and\n'
     '   its partitioning with height).'),

    'altitude':
    ('Defined as the altitude of radar or lidar - the one that is lower.'),

    'Z':
    ('This variable has been corrected for attenuation by gaseous\n'
     'attenuation (using the thermodynamic variables from a forecast\n'
     'model; see the radar_gas_atten variable) and liquid attenuation\n'
     '(using liquid water path from a microwave radiometer; see the\n'
     'radar_liquid_atten variable) but rain and melting-layer attenuation\n'
     'has not been corrected. Calibration convention: in the absence of\n'
     'attenuation, a cloud at 273 K containing one million 100-micron droplets\n'
     'per cubic metre will have a reflectivity of 0 dBZ at all frequencies.'),

    'bias':
    ('This variable is an estimate of the one-standard-deviation calibration error.'),

    'ldr':
    ('This parameter is the ratio of cross-polar to co-polar reflectivity.'),

    'width':
    ('This parameter is the standard deviation of the reflectivity-weighted\n'
     'velocities in the radar pulse volume.'),

    'v':
    ('This parameter is the radial component of the velocity, with positive\n'
     'velocities are away from the radar.'),
}
