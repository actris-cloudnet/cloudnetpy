"""Initial Metadata of Cloudnet variables for NetCDF file writing.
"""

from collections import namedtuple

FIELDS = (
    'long_name',
    'units',
    'plot_range',
    'plot_scale',
    'comment',
    'definition',
    'references',
    'ancillary_variables',
    'sensitivity_variable',
    'positive')

MetaData = namedtuple('MetaData', FIELDS, defaults=(None,)*len(FIELDS))

_LOG = 'logarithmic'
_LIN = 'linear'

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
     '       be aware that errors in reflectivity may result.'),

    'classification_pixels':
    ('\nValue 0: Clear sky.\n'
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

    'classification_quality_pixels':
    ('\nValue 0: Clear sky.\n'
     'Value 1: Lidar echo only.\n'
     'Value 2: Radar echo but reflectivity may be unreliable as attenuation by rain,\n'
     '         melting ice or liquid cloud has not been corrected.\n'
     'Value 3: Good radar and lidar echos.\n'
     'Value 4: No radar echo but rain or liquid cloud beneath mean that attenuation\n'
     '         that would be experienced is unknown.\n'
     'Value 5: Good radar echo only.\n'
     'Value 6: No radar echo but known attenuation.\n'
     'Value 7: Radar echo corrected for liquid cloud attenuation\n'
     '         using microwave radiometer data.\n'
     'Value 8: Radar ground clutter.\n'
     'Value 9: Lidar clear-air molecular scattering.'),

    'iwc_retrieval_status':
    ('\n0: No ice present\n'
      '1: Reliable retrieval\n'
      '2: Unreliable retrieval due to uncorrected attenuation from liquid water below the ice (no liquid water path measurement available)\n'
      '3: Retrieval performed but radar corrected for liquid attenuation using radiometer liquid water path which is not always accurate\n'
      '4: Ice detected only by the lidar\n'
      '5: Ice detected by radar but rain below so no retrieval performed due to very uncertain attenuation\n'
      '6: Clear sky above rain, wet-bulb temperature less than 0degC: if rain attenuation were strong then ice could be present but undetected\n'
      '7: Drizzle or rain that would have been classified as ice if the wet-bulb temperature were less than 0degC: may be ice if temperature is in error'),

    'lwc_retrieval_status':
    ('\n0: No liquid water detected\n'
      '1: Reliable retrieval \n'
      '2: Adiabatic retrieval where cloud top has been adjusted to match liquid water path from microwave radiometer because layer is not detected by radar\n'
      '3: Adiabatic retrieval: new cloud pixels where cloud top has been adjusted to match liquid water path from microwave radiometer because layer is not detected by radar\n'
      '4: No retrieval: either no liquid water path is available or liquid water path is uncertain\n'
      '5: No retrieval: liquid water layer detected only by the lidar and liquid water path is unavailable or uncertain:\n'
      '   cloud top may be higher than diagnosed cloud top since lidar signal has been attenuated\n'
      '6: Rain present: cloud extent is difficult to ascertain and liquid water path also uncertain.'),

    'model_number':
        ('\n0: Single polarisation radar.'
         '\n1: Dual polarisation radar.'),

    'dual_polarization':
        ('\n0: Single polarisation radar.'
         '\n1: Dual polarisation radar in linear depolarisation ratio (LDR) mode.'
         '\n2: Dual polarisation radar in simultaneous transmission simultaneous reception (STSR) mode.'),

    'FFT_window':
        ('\n0: square'
         '\n1: parzen'
         '\n2: blackman'
         '\n3: welch'
         '\n4: slepian2'
         '\n5: slepian3'),

    'quality_flag':
        ('\nBit 0: ADC saturation.'
         '\nBit 1: Spectral width too high.'
         '\nBit 2: No transmission power levelling.')

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

    'classification_pixels':
    ('This variable is a simplification of the bitfield "category_bits" in the target categorization\n'
     'and data quality dataset. It provides the 9 main atmospheric target classifications\n'
     'that can be distinguished by radar and lidar.\n'
     'The classes are defined in the definition attributes.'),

    'classification_quality_pixels':
    ('This variable is a simplification of the bitfield "quality_bits"\n'
     'in the target categorization and data quality dataset.\n'
     'It reports on the reliability of the radar and lidar data used to perform the classification.\n'
     'The classes are defined in the definition attributes.'),

    'cloud_mask':
    ('This variable was calculated from the instance of cloud in the cloud mask variable\n'
     'and provides array of total cloud layer.'),

    'cloud_bottom':
    ('This variable was calculated from the instance of cloud in the cloud mask variable\n'
     'and provides cloud base height for a maximum of 1 cloud layers.'),

    'cloud_top':
    ('This variable was calculated from the instance of cloud in the cloud mask variable\n'
     'and provides cloud base top for a maximum of 1 cloud layers.'),

    'iwc':
    ('This variable was calculated from the x-GHz radar reflectivity factor after correction for gaseous attenuation,\n'
    'and temperature taken from a forecast model, using the following empirical formula:\n'
    'log10(iwc[g m-3]) = 0.00058Z[dBZ]T[degC] + 0.0923Z[dBZ] + -0.00706T[degC] + -0.992.\n'
    'In this formula Z is taken to be defined such that all frequencies of radar would measure the same Z in Rayleigh scattering ice.\n'
    'However, the radar is more likely to have been calibrated such that all frequencies would measure the same Z in Rayleigh scattering\n'
    'liquid cloud at 0 degrees C. The measured Z is therefore multiplied by |K(liquid,0degC,94GHz)|^2/0.93 = 0.7194 before applying this formula.\n'
    'The formula has been used where the \"categorization\" data has diagnosed that the radar echo is due to ice, but note that in some cases\n'
    'supercooled drizzle will erroneously be identified as ice. Missing data indicates either that ice cloud was present but it was only\n'
    'detected by the lidar so its ice water content could not be estimated, or that there was rain below the ice associated with uncertain\n'
    'attenuation of the reflectivities in the ice.\n'
    'Note that where microwave radiometer liquid water path was available it was used to correct the radar for liquid attenuation when liquid\n'
    'cloud occurred below the ice; this is indicated a value of 3 in the iwc_retrieval_status variable.  There is some uncertainty in this\n'
    'prodedure which is reflected by an increase in the associated values in the iwc_error variable.\n'
    'When microwave radiometer data were not available and liquid cloud occurred below the ice, the retrieval was still performed but its\n'
    'reliability is questionable due to the uncorrected liquid water attenuation. This is indicated by a value of 2 in the iwc_retrieval_status\n'
    'variable, and an increase in the value of the iwc_error variable'),

    'iwc_error':
    ('This variable is an estimate of the one-standard-deviation random error in ice water content\n'
    'due to both the uncertainty of the retrieval (about +50%/-33%, or 1.7 dB), and the random error in radar reflectivity factor from which ice water\n'
    'content was calculated. When liquid water is present beneath the ice but no microwave radiometer data were available to correct for the\n'
    'associated attenuation, the error also includes a contribution equivalent to approximately 250 g m-2 of liquid water path being uncorrected for.\n'
    'As uncorrected liquid attenuation actually results in a systematic underestimate of ice water content, users may wish to reject affected data;\n'
    'these pixels may be identified by a value of 2 in the iwc_retrieval_status variable.\n'
    'Typical errors in temperature contribute much less to the overall uncertainty in retrieved ice water content so are not considered.\n'
    'Missing data in iwc_error indicates either zero ice water content (for which an error in dB would be meaningless), or no ice water content value being reported.\n'
    'Note that when zero ice water content is reported, it is possible that ice cloud was present but was just not detected by any of the instruments.'),

    'iwc_bias':
    ('This variable was calculated from the instance of cloud in the cloud mask variable\n'
     'and provides cloud base top for a maximum of 1 cloud layers.'),

    'iwc_sensitivity':
    ('This variable is an estimate of the minimum detectable ice water content as a function of height.'),

    'iwc_retrieval_status':
    ('This variable describes whether a retrieval was performed for each pixel, and its associated quality, in the form of 8 different classes.\n'
    'The classes are defined in the definition and long_definition attributes. The most reliable retrieval is that without any rain or liquid\n'
    'cloud beneath, indicated by the value 1, then the next most reliable is when liquid water attenuation has been corrected using a microwave\n'
    'radiometer, indicated by the value 3, while a value 2 indicates that liquid water cloud was present but microwave radiometer data were not\n'
    'available so no correction was performed. No attempt is made to retrieve ice water content when rain is present below the ice; this is\n'
    'indicated by the value 5.'),

    'iwc_inc_rain':
    ('This variable is the same as iwc, \n'
     'except that values of iwc in ice above rain have been included. \n'
     'This variable contains values \n'
     'which have been severely affected by attenuation \n'
     'and should only be used when the effect of attenuation is being studied.'),

    'lwc':
    ('This variable was calculated for the profiles where the \"categorization\" data has diagnosed that liquid water is present \n'
    'and liquid water path is available from a coincident microwave radiometer. The model temperature and pressure were used to estimate the \n'
    'theoretical adiabatic liquid water content gradient for each cloud base and the adiabatic liquid water content is then scaled so that its \n'
    'integral matches the radiometer measurement so that the liquid water content now follows a quasi-adiabatic profile. If the liquid layer is \n'
    'detected by the lidar only, there is the potential for cloud top height to be underestimated and so if the adiabatic integrated liquid water \n'
    'content is less than that measured by the microwave radiometer, the cloud top is extended until the adiabatic integrated liquid water content \n'
    'agrees with the value measured by the microwave radiometer. \n'
    'Missing values indicate that either 1) a liquid water layer was diagnosed but no microwave radiometer data was available, \n'
    '2) a liquid water layer was diagnosed but the microwave radiometer data was unreliable; this may be because a melting layer was present in the profile, \n'
    'or because the retrieved lwp was unphysical (values of zero are not uncommon for thin supercooled liquid layers), or \n'
    '3) that rain is present in the profile and therefore, the vertical extent of liquid layers is difficult to ascertain. '),

    'lwc_error':
    ('This variable is an estimate of the random error in liquid water content due to the uncertainty in the microwave radiometer \n'
    'liquid water path retrieval and the uncertainty in cloud base and/or cloud top height. This is associated with the resolution of the grid used, 20 m, \n'
    'which can affect both cloud base and cloud top. If the liquid layer is detected by the lidar only, there is the potential for cloud top height to be underestimated. \n'
    'Similarly, there is the possibility that the lidar may not detect the second cloud base when multiple layers are present and the cloud base will be \n'
    'overestimated. It is assumed that the error contribution arising from using the model temperature and pressure at cloud base is negligible.'),

    'lwc_retrieval_status':
    ('This variable describes whether a retrieval was performed for each pixel, and its associated quality, in the form of 6 different classes.\n'
    'The classes are defined in the definition and long_definition attributes. The most reliable retrieval is that when both radar and lidar detect the liquid layer, and \n'
    'microwave radiometer data is present, indicated by the value 1. The next most reliable is when microwave radiometer data is used to adjust the cloud depth when \n'
    'the radar does not detect the liquid layer, indicated by the value 2, with a value of 3 indicating the cloud pixels that have been added at cloud top to avoid the \n'
    'profile becoming superadiabatic. A value of 4 indicates that microwave radiometer data were not available or not reliable (melting level present or unphysical values) \n'
    'but the liquid layers were well defined.  If cloud top was not well defined then this is indicated by a value of 5. \n'
    'The full retrieval of liquid water content, which requires reliable liquid water path from the microwave radiometer, was only performed for classes 1-3. \n'
    'No attempt is made to retrieve liquid water content when rain is present; this is indicated by the value 6.'),

    'LWP':
    ('This variable is the vertically integrated liquid water directly over the site.\n'
    'The temporal correlation of errors in liquid water path means that it is not really meaningful to distinguish bias from random error,\n'
    'so only an error variable is provided.\n'
    'Original comment: These values denote the vertically integrated amount of condensed water from the surface to TOA.'),

    'LWP_error':
    ('This variable is a rough estimate of the one-standard-deviation error in liquid water path, calculated as a'
    'combination of a 20 g m-2 linear error and a 25% fractional error.'),

    'lwc_th':
    ('This variable is the liquid water content assuming a tophat distribution.'
    'I.e. the profile of liquid water content in each layer is constant.'),

    'radar_liquid_atten':
    ('This variable was calculated from the liquid water path\n'
     'measured by microwave radiometer using lidar and radar returns to perform\n'
     'an approximate partitioning of the liquid water content with height. Bit 5 of\n'
     'the quality_bits variable indicates where a correction for liquid water\n'
     'attenuation has been performed.'),

    'radar_gas_atten':
    ('This variable was calculated from the model temperature,\n'
     'pressure and humidity, but forcing pixels containing liquid cloud to saturation\n'
     'with respect to liquid water. It was calculated using the millimeter-wave propagation\n'
     'model of Liebe (1985, Radio Sci. 20(5), 1069-1089). It has been used to correct Z.'),

    'Tw':
    ('This variable was calculated from model T, P and relative humidity, first\n'
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
    'lidar_wavelength': MetaData(
        'Laser wavelength',
        'nm'
    ),
    'ldr': MetaData(
        'Linear depolarisation ratio',
        'dB',
        (-30, 0),
        _LIN,
        comment=_COMMENTS['ldr']
    ),
    'width': MetaData(
        'Spectral width',
        'm s-1',
        (0, 3),
        _LOG,
        comment=_COMMENTS['width']
    ),
    'v': MetaData(
        'Doppler velocity',
        'm s-1',
        (-4, 2),
        _LIN,
        comment=_COMMENTS['v'],
        positive='up',
    ),
    'SNR': MetaData(
        'Signal-to-noise ratio',
        'dB',
        (-20, 60),
        _LIN
    ),
    'Z': MetaData(
        'Radar reflectivity factor',
        'dBZ',
        (-40, 20),
        _LIN,
        comment=_COMMENTS['Z'],
        ancillary_variables='Z_error Z_bias Z_sensitivity'
    ),
    'Z_error': MetaData(
        'Error in radar reflectivity factor',
        'dB',
        comment=_COMMENTS['Z_error']
    ),
    'Z_bias': MetaData(
        'Bias in radar reflectivity factor',
        'dB',
        comment=_COMMENTS['bias']
    ),
    'Z_sensitivity': MetaData(
        'Minimum detectable radar reflectivity',
        'dBZ',
        comment=_COMMENTS['Z_sensitivity']
    ),
    'Zh': MetaData(
        'Radar reflectivity factor (uncorrected), horizontal polarization',
        'dBZ',
        (-40, 20),
        _LIN
    ),
    'radar_liquid_atten': MetaData(
        'Approximate two-way radar attenuation due to liquid water',
        'dB',
        (0, 10),
        _LIN,
        comment=_COMMENTS['radar_liquid_atten']
    ),
    'radar_gas_atten': MetaData(
        'Two-way radar attenuation due to atmospheric gases',
        'dB',
        (0, 4),
        _LIN,
        comment=_COMMENTS['radar_gas_atten']
    ),
    'Tw': MetaData(
        'Wet-bulb temperature',
        'K',
        (200, 300),
        _LIN,
        comment=_COMMENTS['Tw']
    ),
    'vwind': MetaData(
        'Meridional wind',
        'm s-1',        
        (-50, 50),
        _LIN
    ),
    'uwind': MetaData(
        'Zonal wind',
        'm s-1',
        (-50, 50),
        _LIN
    ),
    'q': MetaData(
        'Specific humidity',
        '',
        (0, 0.2),
        _LIN
    ),
    'temperature': MetaData(
        'Temperature',
        'K',
        (200, 300),
        _LIN
    ),
    'pressure': MetaData(
        'Pressure',
        'Pa',
        (0, 110000),
        _LIN
    ),
    'beta': MetaData(
        'Attenuated backscatter coefficient',
        'sr-1 m-1',
        (1e-7, 1e-4),
        _LOG,
        ancillary_variables='beta_error beta_bias'
    ),
    'beta_raw': MetaData(
        'Raw attenuated backscatter coefficient',
        'sr-1 m-1',
        (1e-7, 1e-4),
        _LOG,
    ),
    'beta_error': MetaData(
        'Error in attenuated backscatter coefficient',
        'dB',
    ),
    'beta_bias': MetaData(
        'Bias in attenuated backscatter coefficient',
        'dB',
    ),
    'lwp': MetaData(
        'Liquid water path',
        'g m-2',
        (-100, 1000),
        _LIN
    ),
    'lwp_error': MetaData(
        'Error in liquid water path',
        'g m-2',
    ),
    'category_bits': MetaData(
        'Target categorization bits',
        comment=_COMMENTS['category_bits'],
        definition=_DEFINITIONS['category_bits']
    ),
    'quality_bits': MetaData(
        'Data quality bits',
        comment=_COMMENTS['quality_bits'],
        definition=_DEFINITIONS['quality_bits']
    ),
    # product variables
    'target_classification': MetaData(
        'Target classification',
        comment=_COMMENTS['classification_pixels'],
        definition=_DEFINITIONS['classification_pixels']
    ),
    'detection_status': MetaData(
        'Radar and lidar detection status',
        comment=_COMMENTS['classification_quality_pixels'],
        definition=_DEFINITIONS['classification_quality_pixels']
    ),
    'cloud_mask': MetaData(
        'Total area of clouds',
        comment=_COMMENTS['cloud_mask'],
    ),
    'cloud_bottom': MetaData(
        'Height of cloud base above ground',
        'm',
        comment=_COMMENTS['cloud_bottom'],
    ),
    'cloud_top': MetaData(
        'Height of cloud top above ground',
        'm',
        comment=_COMMENTS['cloud_top'],
    ),
    'iwc': MetaData(
        'Ice water content',
        'kg m-3',
        (1e-7, 0.001),
        _LOG,
        comment=_COMMENTS['iwc'],
        sensitivity_variable='iwc_sensitivity'
    ),
    'iwc_error': MetaData(
        'Random error in ice water content, one standard deviation',
        'dB',
        (0, 3),
        _LIN,
        comment=_COMMENTS['iwc_error']
    ),
    'iwc_bias': MetaData(
        'Possible bias in ice water content, one standard deviation',
        'dB',
        comment=_COMMENTS['iwc_bias']
    ),
    'iwc_sensitivity': MetaData(
        'Minimum detectable ice water content',
        'kg m-3',
        comment=_COMMENTS['iwc_sensitivity']
    ),
    'iwc_retrieval_status': MetaData(
        'Ice water content retrieval status',
        comment=_COMMENTS['iwc_retrieval_status'],
        definition=_DEFINITIONS['iwc_retrieval_status']
    ),
    'iwc_inc_rain': MetaData(
        'Ice water content',
        'kg m-3',
        (1e-7, 0.001),
        _LOG,
        comment=_COMMENTS['iwc_inc_rain'],
        sensitivity_variable='iwc_sensitivity'
    ),
    'lwc': MetaData(
        'Liquid water content',
        'kg m-3',
        (1e-5, 0.01),
        _LOG,
        comment=_COMMENTS['lwc']
    ),
    'lwc_error': MetaData(
        'Random error in liquid water content, one standard deviation',
        'kg m-3',
        comment=_COMMENTS['lwc_error'],
    ),
    'lwc_retrieval_status': MetaData(
        'Liquid water content retrieval status',
        'scalar',
        (0,6),
        comment=_COMMENTS['lwc_retrieval_status'],
        definition=_DEFINITIONS['lwc_retrieval_status']
    ),
    'LWP': MetaData(
        'Liquid water path',
        'kg m-2',
        (-100, 1000),
        _LIN,
        comment=_COMMENTS['LWP']
    ),
    'LWP_error': MetaData(
        'Random error in liquid water path, one standard deviation',
        'kg m-2',
        comment=_COMMENTS['LWP_error']
    ),
    'lwc_th': MetaData(
        'Liquid water content (tophat distribution)',
        'kg m-3',
        comment=_COMMENTS['lwc_th']
    ),
    'insect_prob': MetaData(
        'Insect probability',
        '',
        (0, 1),
        _LIN
    ),
    # RPG variables:
    'Ze': MetaData(
        'Radar reflectivity factor (uncorrected), vertical polarization',
        'dBZ',
        (-40, 20),
        _LIN
    ),
    'rain_rate': MetaData(
        'Rain rate',
        'mm h-1',
    ),
    'input_voltage_range': MetaData(
        'ADC input voltage range (+/-)',
        'mV',
    ),
    'noise_threshold': MetaData(
        'Noise filter threshold factor',
        '',
        comment='Multiple of the standard deviation of Doppler spectra.'
    ),
    'antenna_separation': MetaData(
        'Antenna separation',
        'm',
    ),
    'antenna_diameter': MetaData(
        'Antenna diameter',
        'm',
    ),
    'antenna_gain': MetaData(
        'Antenna gain',
        'dB',
    ),
    'range_resolution': MetaData(
        'Vertical resolution of range',
        'm',
    ),
    'half_power_beam_width': MetaData(
        'Half power beam width',
        'degrees',
    ),
    'transmitter_temperature': MetaData(
        'Transmitter temperature',
        'K',
    ),
    'transmitted_power': MetaData(
        'Transmitted power',
        'W',
    ),
    'number_of_spectral_samples': MetaData(
        'Number of spectral samples in each chirp sequence',
        '',
    ),
    'skewness': MetaData(
        'Skewness of spectra',
        '',
    ),
    'kurtosis': MetaData(
        'Kurtosis of spectra',
    ),
    'azimuth': MetaData(
        'Azimuth angle',
        'degrees',
    ),
    'elevation': MetaData(
        'Elevation angle above horizon',
        'degrees',
    ),
    'if_power': MetaData(
        'IF power at ACD',
        'uW',
    ),
    'brightness_temperature': MetaData(
        'Brightness temperature',
        'K',
    ),
    'voltage': MetaData(
        'Voltage',
        'V',
    ),
    'wind_direction': MetaData(
        'Wind direction',
        'degrees',
    ),
    'wind_speed': MetaData(
        'Wind speed',
        'm s-1',
    ),
    'pc_temperature': MetaData(
        'PC temperature',
        'K',
    ),
    'receiver_temperature': MetaData(
        'Receiver temperature',
        'K',
    ),
    'time_ms': MetaData(
        'Time ms',
        'ms',
    ),
    'integration_time': MetaData(
        'Integration time',
        's',
        comment='Effective integration time of chirp sequence',
    ),
    'file_code': MetaData(
        'File code',
        comment='Indicates the RPG software version.',
    ),
    'program_number': MetaData(
        'Program number',
    ),
    'model_number': MetaData(
        'Model number',
        definition=_DEFINITIONS['model_number']
    ),
    'sample_duration': MetaData(
        'Sample duration',
        's'
    ),
    'dual_polarization': MetaData(
        'Dual polarisation type',
        definition=_DEFINITIONS['dual_polarization']
    ),
    'number_of_averaged_chirps': MetaData(
        'Number of averaged chirps in sequence'
    ),
    'chirp_start_indices': MetaData(
        'Chirp sequences start indices'
    ),
    'calibration_interval': MetaData(
        'Calibration interval in samples'
    ),
    'status_flag': MetaData(
        'Status flag for heater and blower'
    ),
    'FFT_window': MetaData(
        'FFT window type',
        definition=_DEFINITIONS['FFT_window']
    ),
    'quality_flag': MetaData(
        'Quality flag',
        definition=_DEFINITIONS['quality_flag']
    ),
    'nyquist_velocity': MetaData(
        'Nyquist velocity',
        'm s-1'
    ),
    'correlation_coefficient': MetaData(
        'Correlation coefficient',
    ),
    'Zdr': MetaData(
        'Differential reflectivity',
        'dB'
    ),
    'spectral_differential_phase': MetaData(
        'Spectral differential phase'),
}
