"""Initial Metadata of Cloudnet variables for NetCDF file writing.
"""

from collections import namedtuple

FIELDS = (
    'long_name',
    'units',
    'comment',
    'definition',
    'references',
    'ancillary_variables',
    'positive')

MetaData = namedtuple('MetaData', FIELDS, defaults=(None,)*len(FIELDS))

COMMENTS = {
    'ldr':
        'This parameter is the ratio of cross-polar to co-polar reflectivity.',

    'width':
        ('This parameter is the standard deviation of the reflectivity-weighted\n'
         'velocities in the radar pulse volume.'),

    'v':
        ('This parameter is the radial component of the velocity, with positive\n'
         'velocities are away from the radar.'),

}

DEFINITIONS = {
    'model_number':
        ('\n'
         '0: Single polarisation radar.\n'
         '1: Dual polarisation radar.'),

    'dual_polarization':
        ('\n'
         'Value 0: Single polarisation radar.\n'
         'Value 1: Dual polarisation radar in linear depolarisation ratio (LDR)\n'
         '         mode.\n'
         'Value 2: Dual polarisation radar in simultaneous transmission\n'
         '         simultaneous reception (STSR) mode.'),

    'FFT_window':
        ('\n'
         'Value 0: Square\n'
         'Value 1: Parzen\n'
         'Value 2: Blackman\n'
         'Value 3: Welch\n'
         'Value 4: Slepian2\n'
         'Value 5: Slepian3'),

    'quality_flag':
        ('\n'
         'Bit 0: ADC saturation.\n'
         'Bit 1: Spectral width too high.\n'
         'Bit 2: No transmission power levelling.')

}


COMMON_ATTRIBUTES = {
    'time': MetaData(
        long_name='Time UTC',
        units='decimal hours since midnight'
    ),
    'height': MetaData(
        long_name='Height above mean sea level',
        units='m',
        comment='Height grid from the mean sea level towards zenith.'
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
        units='degrees_north'
    ),
    'altitude': MetaData(
        long_name='Altitude of site',
        units='m'
    ),
    'range': MetaData(
        long_name='Range from instrument',
        units='m',
        comment='Height grid from the instrument towards the line of sight.'
    ),

    'Ze': MetaData(
        long_name='Radar reflectivity factor (uncorrected), vertical polarization',
        units='dBZ',
    ),
    'width': MetaData(
        long_name='Spectral width',
        units='m s-1',
        comment=COMMENTS['width']
    ),
    'v': MetaData(
        long_name='Doppler velocity',
        units='m s-1',
        comment=COMMENTS['v'],
        positive='up',
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
        comment=COMMENTS['ldr']
    ),
    'lwp': MetaData(
        long_name='Liquid water path',
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
    )
}

RPG_ATTRIBUTES = {
    'file_code': MetaData(
        long_name='File code',
        comment='Indicates the RPG software version.',
    ),
    'program_number': MetaData(
        long_name='Program number',
    ),
    'model_number': MetaData(
        long_name='Model number',
        definition=DEFINITIONS['model_number']
    ),
    'antenna_separation': MetaData(
        long_name='Antenna separation',
        units='m',
    ),
    'antenna_diameter': MetaData(
        long_name='Antenna diameter',
        units='m',
    ),
    'antenna_gain': MetaData(
        long_name='Antenna gain',
        units='dB',
    ),
    'half_power_beam_width': MetaData(
        long_name='Half power beam width',
        units='degrees',
    ),
    'dual_polarization': MetaData(
        long_name='Dual polarisation type',
        definition=DEFINITIONS['dual_polarization']
    ),
    'sample_duration': MetaData(
        long_name='Sample duration',
        units='s'
    ),
    'calibration_interval': MetaData(
        long_name='Calibration interval in samples'
    ),
    'number_of_spectral_samples': MetaData(
        long_name='Number of spectral samples in each chirp sequence',
        units='',
    ),
    'chirp_start_indices': MetaData(
        long_name='Chirp sequences start indices'
    ),
    'number_of_averaged_chirps': MetaData(
        long_name='Number of averaged chirps in sequence'
    ),
    'integration_time': MetaData(
        long_name='Integration time',
        units='s',
        comment='Effective integration time of chirp sequence',
    ),
    'range_resolution': MetaData(
        long_name='Vertical resolution of range',
        units='m',
    ),
    'FFT_window': MetaData(
        long_name='FFT window type',
        definition=DEFINITIONS['FFT_window']
    ),
    'input_voltage_range': MetaData(
        long_name='ADC input voltage range (+/-)',
        units='mV',
    ),
    'noise_threshold': MetaData(
        long_name='Noise filter threshold factor',
        units='',
        comment='Multiple of the standard deviation of Doppler spectra.'
    ),
    'time_ms': MetaData(
        long_name='Time ms',
        units='ms',
    ),
    'quality_flag': MetaData(
        long_name='Quality flag',
        definition=DEFINITIONS['quality_flag']
    ),
    'voltage': MetaData(
        long_name='Voltage',
        units='V',
    ),
    'brightness_temperature': MetaData(
        long_name='Brightness temperature',
        units='K',
    ),
    'if_power': MetaData(
        long_name='IF power at ACD',
        units='uW',
    ),
    'elevation': MetaData(
        long_name='Elevation angle above horizon',
        units='degrees',
    ),
    'azimuth': MetaData(
        long_name='Azimuth angle',
        units='degrees',
    ),
    'status_flag': MetaData(
        long_name='Status flag for heater and blower'
    ),
    'transmitted_power': MetaData(
        long_name='Transmitted power',
        units='W',
    ),
    'transmitter_temperature': MetaData(
        long_name='Transmitter temperature',
        units='K',
    ),
    'receiver_temperature': MetaData(
        long_name='Receiver temperature',
        units='K',
    ),
    'pc_temperature': MetaData(
        long_name='PC temperature',
        units='K',
    ),
    'skewness': MetaData(
        long_name='Skewness of spectra',
        units='',
    ),
    'correlation_coefficient': MetaData(
        long_name='Correlation coefficient',
    ),
    'spectral_differential_phase': MetaData(
        long_name='Spectral differential phase'
    ),
    'wind_direction': MetaData(
        long_name='Wind direction',
        units='degrees',
    ),
    'wind_speed': MetaData(
        long_name='Wind speed',
        units='m s-1',
    )
}

MIRA_ATTRIBUTES = {
    'Zdr': MetaData(
        long_name='Differential reflectivity',
        units='dB'
    ),
    'SNR': MetaData(
        long_name='Signal-to-noise ratio',
        units='dB',
    )
}
