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

_DEFINITIONS = {
    'category_bits':
    ('\n'
     'Bit 0: Small liquid droplets are present.\n'
     'Bit 1: Falling hydrometeors are present; if Bit 2 is set then these are most\n'
     '       likely ice particles, otherwise they are drizzle or rain drops.\n'
     'Bit 2: Wet-bulb temperature is less than 0 degrees C, implying\n'
     '       the phase of Bit-1 particles.\n'
     'Bit 3: Melting ice particles are present.\n'
     'Bit 4: Aerosol particles are present and visible to the lidar.\n'
     'Bit 5: Insects are present and visible to the radar.'),

    'quality_bits':
    ('\n'
     'Bit 0: An echo is detected by the radar.\n'
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
     '       be aware that errors in reflectivity may result.'),

    'target_classification':
    ('\n'
     'Value 0: Clear sky.\n'
     'Value 1: Cloud liquid droplets only.\n'
     'Value 2: Drizzle or rain.\n'
     'Value 3: Drizzle or rain coexisting with cloud liquid droplets.\n'
     'Value 4: Ice particles.\n'
     'Value 5: Ice coexisting with supercooled liquid droplets.\n'
     'Value 6: Melting ice particles.\n'
     'Value 7: Melting ice particles coexisting with cloud liquid droplets.\n'
     'Value 8: Aerosol particles, no cloud or precipitation.\n'
     'Value 9: Insects, no cloud or precipitation.\n'
     'Value 10: Aerosol coexisting with insects, no cloud or precipitation.'),

    'detection_status':
    ('\n'
     'Value 0: Clear sky.\n'
     'Value 1: Lidar echo only.\n'
     'Value 2: Radar echo but reflectivity may be unreliable as attenuation by\n'
     '         rain, melting ice or liquid cloud has not been corrected.\n'
     'Value 3: Good radar and lidar echos.\n'
     'Value 4: No radar echo but rain or liquid cloud beneath mean that\n'
     '         attenuation that would be experienced is unknown.\n'
     'Value 5: Good radar echo only.\n'
     'Value 6: No radar echo but known attenuation.\n'
     'Value 7: Radar echo corrected for liquid cloud attenuation\n'
     '         using microwave radiometer data.\n'
     'Value 8: Radar ground clutter.\n'
     'Value 9: Lidar clear-air molecular scattering.'),

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
     'attribute. Bit 0 is the least significant.'),

    'target_classification':
    ('This variable is a simplification of the bitfield "category_bits" in the\n'
     'target categorization and data quality dataset. It provides the 9 main\n'
     'atmospheric target classifications that can be distinguished by radar and\n'
     'lidar. The classes are defined in the definition attributes.'),

    'detection_status':
    ('This variable is a simplification of the bitfield "quality_bits" in the\n'
     'target categorization and data quality dataset. It reports on the\n'
     'reliability of the radar and lidar data used to perform the classification.\n'
     'The classes are defined in the definition attributes.'),

    'cloud_mask':
    ('This variable was calculated from the instance of cloud in the cloud mask\n'
     'variable and provides array of total cloud layer.'),

    'cloud_bottom':
    ('This variable was calculated from the instance of cloud in the cloud mask\n'
     'variable and provides cloud base height for a maximum of 1 cloud layers.'),

    'cloud_top':
    ('This variable was calculated from the instance of cloud in the cloud mask\n'
     'variable and provides cloud base top for a maximum of 1 cloud layers.'),

    'LWP':
    ('This variable is the vertically integrated liquid water directly over the\n'
     'site. The temporal correlation of errors in liquid water path means that\n'
     'it is not really meaningful to distinguish bias from random error, so only\n'
     'an error variable is provided.'),

    'LWP_error':
    ('This variable is a rough estimate of the one-standard-deviation error\n'
     'in liquid water path, calculated as a combination of a 20 g m-2 linear\n'
     'error and a 25% fractional error.'),

    'lwc_th':
    ('This variable is the liquid water content assuming a tophat distribution.\n'
     'I.e. the profile of liquid water content in each layer is constant.'),

    'radar_liquid_atten':
    ('This variable was calculated from the liquid water path measured by\n'
     'microwave radiometer using lidar and radar returns to perform an \n'
     'approximate partitioning of the liquid water content with height.\n'
     'Bit 5 of the quality_bits variable indicates where a correction for\n'
     'liquid water attenuation has been performed.'),

    'radar_gas_atten':
    ('This variable was calculated from the model temperature, pressure and\n'
     'humidity, but forcing pixels containing liquid cloud to saturation with\n'
     'respect to liquid water. It has been used to correct Z.'),

    'Tw':
    ('This variable was calculated from model T, P and relative humidity, first\n'
     'interpolated into measurement grid.'),

    'Z_sensitivity':
    ('This variable is an estimate of the radar sensitivity, i.e. the minimum\n'
     'detectable radar reflectivity, as a function of height. It includes the\n'
     'effect of ground clutter and gas attenuation but not liquid attenuation.'),

    'Z_error':
    ('This variable is an estimate of the one-standard-deviation random error in\n'
     'radar reflectivity factor. It originates from the following independent\n'
     'sources of error:\n'
     '1) Precision in reflectivity estimate due to finite signal to noise\n'
     '   and finite number of pulses\n'
     '2) 10% uncertainty in gaseous attenuation correction (mainly due to\n'
     '   error in model humidity field)\n'
     '3) Error in liquid water path (given by the variable lwp_error) and\n'
     '   its partitioning with height).'),

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
    'This variable is an estimate of the one-standard-deviation calibration error.',

    'ldr':
    'This parameter is the ratio of cross-polar to co-polar reflectivity.',

    'width':
    ('This parameter is the standard deviation of the reflectivity-weighted\n'
     'velocities in the radar pulse volume.'),

    'v':
    ('This parameter is the radial component of the velocity, with positive\n'
     'velocities are away from the radar.'),
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
        comment=_COMMENTS['width']
    ),
    'v': MetaData(
        long_name='Doppler velocity',
        units='m s-1',
        comment=_COMMENTS['v'],
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
        comment=_COMMENTS['ldr']
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

CLASSIFICATION_ATTRIBUTES = {
    'target_classification': MetaData(
        long_name='Target classification',
        comment=_COMMENTS['target_classification'],
        definition=_DEFINITIONS['target_classification']
    ),
    'detection_status': MetaData(
        long_name='Radar and lidar detection status',
        comment=_COMMENTS['detection_status'],
        definition=_DEFINITIONS['detection_status']
    )
}

CATEGORIZE_ATTRIBUTES = {
    'Z': MetaData(
        long_name='Radar reflectivity factor',
        units='dBZ',
        comment=_COMMENTS['Z'],
        ancillary_variables='Z_error Z_bias Z_sensitivity'
    ),
    'Z_error': MetaData(
        long_name='Error in radar reflectivity factor',
        units='dB',
        comment=_COMMENTS['Z_error']
    ),
    'Z_bias': MetaData(
        long_name='Bias in radar reflectivity factor',
        units='dB',
        comment=_COMMENTS['bias']
    ),
    'Z_sensitivity': MetaData(
        long_name='Minimum detectable radar reflectivity',
        units='dBZ',
        comment=_COMMENTS['Z_sensitivity']
    ),
    'Zh': MetaData(
        long_name='Radar reflectivity factor (uncorrected), horizontal polarization',
        units='dBZ',
    ),
    'radar_liquid_atten': MetaData(
        long_name='Approximate two-way radar attenuation due to liquid water',
        units='dB',
        comment=_COMMENTS['radar_liquid_atten']
    ),
    'radar_gas_atten': MetaData(
        long_name='Two-way radar attenuation due to atmospheric gases',
        units='dB',
        comment=_COMMENTS['radar_gas_atten'],
        references='Liebe (1985, Radio Sci. 20(5), 1069-1089)'
    ),
    'Tw': MetaData(
        long_name='Wet-bulb temperature',
        units='K',
        comment=_COMMENTS['Tw']
    ),
    'vwind': MetaData(
        long_name='Meridional wind',
        units='m s-1',
    ),
    'uwind': MetaData(
        long_name='Zonal wind',
        units='m s-1',
    ),
    'is_rain': MetaData(
        long_name='Presence of rain',
        comment='Integer denoting the rain (1) or no rain (0).'
    ),
    'beta': MetaData(
        long_name='Attenuated backscatter coefficient',
        units='sr-1 m-1',
        ancillary_variables='beta_error beta_bias'
    ),
    'beta_raw': MetaData(
        long_name='Raw attenuated backscatter coefficient',
        units='sr-1 m-1',
    ),
    'beta_error': MetaData(
        long_name='Error in attenuated backscatter coefficient',
        units='dB',
    ),
    'beta_bias': MetaData(
        long_name='Bias in attenuated backscatter coefficient',
        units='dB',
    ),
    'category_bits': MetaData(
        long_name='Target categorization bits',
        comment=_COMMENTS['category_bits'],
        definition=_DEFINITIONS['category_bits']
    ),
    'quality_bits': MetaData(
        long_name='Data quality bits',
        comment=_COMMENTS['quality_bits'],
        definition=_DEFINITIONS['quality_bits']
    ),
    'insect_prob': MetaData(
        long_name='Insect probability',
        units='',
    ),
    'lidar_wavelength': MetaData(
        long_name='Laser wavelength',
        units='nm'
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
        definition=_DEFINITIONS['model_number']
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
        definition=_DEFINITIONS['dual_polarization']
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
        definition=_DEFINITIONS['FFT_window']
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
        definition=_DEFINITIONS['quality_flag']
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
