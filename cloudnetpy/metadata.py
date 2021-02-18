"""Initial Metadata of Cloudnet variables for NetCDF file writing."""

from collections import namedtuple

FIELDS = (
    'long_name',
    'units',
    'comment',
    'definition',
    'references',
    'ancillary_variables',
    'positive',
    'axis')

MetaData = namedtuple('MetaData', FIELDS)
MetaData.__new__.__defaults__ = (None,) * len(MetaData._fields)


COMMON_ATTRIBUTES = {
    'time': MetaData(
        long_name='Time UTC',
        axis='T',
    ),
    'height': MetaData(
        long_name='Height above mean sea level',
        axis='Z',
        units='m',
        comment='Height grid from the mean sea level towards zenith.'
    ),
    'range': MetaData(
        long_name='Range from instrument',
        axis='Z',
        units='m',
        comment='Height grid from the instrument towards the line of sight.'
    ),
    'model_time': MetaData(
        long_name='Model time UTC',
        units='decimal hours since midnight'
    ),
    'model_height': MetaData(
        long_name='Height of model variables above mean sea level',
        units='m'
    ),
    'latitude': MetaData(
        long_name='Latitude of site',
        units='degrees_north'
    ),
    'longitude': MetaData(
        long_name='Longitude of site',
        units='degrees_east'
    ),
    'altitude': MetaData(
        long_name='Altitude of site',
        units='m'
    ),
    'width': MetaData(
        long_name='Spectral width',
        units='m s-1',
        comment=('This parameter is the standard deviation of the reflectivity-weighted\n'
                 'velocities in the radar pulse volume.'),
    ),
    'v': MetaData(
        long_name='Doppler velocity',
        units='m s-1',
        comment=('This parameter is the radial component of the velocity, with positive\n'
                 'velocities are away from the radar.'),
        positive='up'
    ),
    'v_sigma': MetaData(
        long_name='Standard deviation of mean Doppler velocity',
        units='m s-1'
    ),
    'q': MetaData(
        long_name='Specific humidity',
        units=''
    ),
    'temperature': MetaData(
        long_name='Temperature',
        units='K',
    ),
    'pressure': MetaData(
        long_name='Pressure',
        units='Pa',
    ),
    'ldr': MetaData(
        long_name='Linear depolarisation ratio',
        units='dB',
        comment='This parameter is the ratio of cross-polar to co-polar reflectivity.',
    ),
    'lwp': MetaData(
        long_name='Liquid water path',
        units='',
    ),
    'lwp_error': MetaData(
        long_name='Error in liquid water path',
        units='',
    ),
    'kurtosis': MetaData(
        long_name='Kurtosis of spectra',
        units='',
    ),
    'nyquist_velocity': MetaData(
        long_name='Nyquist velocity',
        units='m s-1'
    ),
    'radar_frequency': MetaData(
        long_name='Radar transmit frequency',
        units='GHz'
    ),
    'rain_rate': MetaData(
        long_name='Rain rate',
        units='mm h-1',
    ),
}
