"""Initial Metadata of Cloudnet variables for NetCDF file writing."""

from collections import namedtuple

FIELDS = (
    'long_name',
    'standard_name',
    'units',
    'comment',
    'definition',
    'references',
    'ancillary_variables',
    'positive',
    'axis',
    'calendar')

MetaData = namedtuple('MetaData', FIELDS)
MetaData.__new__.__defaults__ = (None,) * len(MetaData._fields)


COMMON_ATTRIBUTES = {
    'time': MetaData(
        long_name='Time UTC',
        axis='T',
        standard_name='time',
        calendar='standard'
    ),
    'height': MetaData(
        long_name='Height above mean sea level',
        standard_name='height_above_mean_sea_level',
        units='m',
    ),
    'range': MetaData(
        long_name='Range from instrument',
        axis='Z',
        units='m',
        comment='Distance from instrument to centre of each range bin.'
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
        units='degree_north',
        standard_name='latitude'
    ),
    'longitude': MetaData(
        long_name='Longitude of site',
        units='degree_east',
        standard_name='longitude'
    ),
    'altitude': MetaData(
        long_name='Altitude of site',
        standard_name='altitude',
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
    'beta': MetaData(
        long_name='Attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment='SNR-screened attenuated backscatter coefficient.'
    ),
    'beta_raw': MetaData(
        long_name='Attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment='Non-screened attenuated backscatter coefficient.'
    ),
    'beta_smooth': MetaData(
        long_name='Attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment='SNR-screened attenuated backscatter coefficient.\n'
                'Weak background smoothed using Gaussian 2D-kernel.'
    ),
    'wavelength': MetaData(
        long_name='Laser wavelength',
        units='nm',
    ),
    'zenith_angle': MetaData(
        long_name='Zenith angle',
        units='degree',
        standard_name='zenith_angle',
        comment='Angle to the local vertical. A value of zero is directly overhead.'
    ),
    'Zh': MetaData(
        long_name='Radar reflectivity factor',
        units='dBZ',
        comment='Calibrated reflectivity. Calibration convention: in the absence of attenuation,\n'
                'a cloud at 273 K containing one million 100-micron droplets per cubic metre will\n'
                'have a reflectivity of 0 dBZ at all frequencies.'
    ),
}
