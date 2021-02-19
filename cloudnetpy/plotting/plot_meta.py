"""Metadata for plotting module."""
from collections import namedtuple

FIELDS = ('name',
          'cbar',
          'clabel',
          'ylabel',
          'plot_range',
          'plot_scale',
          'plot_type',
          'source')

PlotMeta = namedtuple('PlotMeta', FIELDS)
PlotMeta.__new__.__defaults__ = (None,) * len(FIELDS)

_LOG = 'logarithmic'
_LIN = 'linear'

_M3 = '$m^{-3}$'
_MS1 = 'm s$^{-1}$'
_SR1M1 = 'sr$^{-1}$ m$^{-1}$'
_KGM2 = 'kg m$^{-2}$'
_KGM3 = 'kg m$^{-3}$'
_KGM2S1 = 'kg m$^{-2}$ s$^{-1}$'
_DB = 'dB'
_DBZ = 'dBZ'

_COLORS = {
    'green': "#3cb371",
    'darkgreen': '#253A24',
    'lightgreen': "#70EB5D",
    'yellowgreen': "#C7FA3A",
    'yellow': "#FFE744",
    'orange': "#ffa500",
    'pink': "#B43757",
    'red': "#F57150",
    'shockred': "#E64A23",
    'seaweed': "#646F5E",
    'seaweed_roll': "#748269",
    'white': "#ffffff",
    'lightblue': "#6CFFEC",
    'blue': "#209FF3",
    'skyblue': "#CDF5F6",
    'darksky': "#76A9AB",
    'darkpurple': "#464AB9",
    'lightpurple': "#6A5ACD",
    'purple': "#BF9AFF",
    'darkgray': "#2f4f4f",
    'lightgray': "#ECECEC",
    'gray': "#d3d3d3",
    'lightbrown': "#CEBC89",
    'lightsteel': "#E5E3EB",
    'steelblue': "#4682b4"
}

# Labels (and corresponding data) starting with an underscore are NOT shown:

_CLABEL = {

    'target_classification':
        (("_Clear sky", _COLORS['white']),
         ("Cloud droplets", _COLORS['lightblue']),
         ("Drizzle or rain", _COLORS['blue']),
         ("Drizzle/rain & cloud droplets", _COLORS['purple']),
         ("Ice", _COLORS['lightsteel']),
         ("Ice & supercooled droplets", _COLORS['darkpurple']),
         ("Melting ice", _COLORS['orange']),
         ("Melting ice & cloud droplets", _COLORS['yellowgreen']),
         ("Aerosols", _COLORS['lightbrown']),
         ("Insects", _COLORS['shockred']),
         ("Aerosols & insects", _COLORS['pink'])),

    'detection_status':
        (("_Clear sky", _COLORS['white']),
         ("Good radar & lidar echos", _COLORS['green']),
         ("Good radar echo only", _COLORS['lightgreen']),
         ("Lidar echo only", _COLORS['yellow']),
         ("Radar corrected for liquid atten.", _COLORS['skyblue']),
         ("Radar uncorrected for liquid atten.", _COLORS['seaweed_roll']),
         ("Radar ground clutter", _COLORS['shockred']),
         ("_Lidar molecular scattering", _COLORS['pink'])),

    'iwc_retrieval_status':
        (("_No ice", _COLORS['white']),
         ("Reliable retrieval", _COLORS['green']),
         ("Corrected liquid attenuation", _COLORS['lightgreen']),
         ("Uncorrected liquid attenuation", _COLORS['orange']),
         ("Ice detected only by lidar", _COLORS['yellow']),
         ("Ice above rain", _COLORS['darksky']),
         ("Clear sky above rain", _COLORS['skyblue']),
         ("Temperature above freezing", _COLORS['gray'])),

    'lwc_retrieval_status':
        (("_No liquid water", _COLORS['white']),
         ("Reliable retrieval", _COLORS['green']),
         ("Adiabatic retrieval: cloud top adjusted", _COLORS['lightgreen']),
         ("Adiabatic retrieval: new cloud pixel", _COLORS['yellow']),
         ("Rain present: no retrieval", _COLORS['lightgray'])),

    'drizzle_retrieval_status':
        (("_No drizzle", _COLORS['white']),
         ("Reliable retrieval", _COLORS['green']),
         ("Retrieval below melting layer", _COLORS['lightgreen']),
         ("Drizzle present but no retrieval possible", _COLORS['red']),
         ("Warm drizzle-free liquid water cloud", _COLORS['orange']),
         ("Rain present: no retrieval", _COLORS['lightgray']))
}

_CBAR = {
    'bit':
        (_COLORS['white'],
         _COLORS['steelblue'])
}

ATTRIBUTES = {
    'Do': PlotMeta(
        name='Drizzle median diameter',
        cbar='viridis',
        clabel='m',
        plot_range=(1e-6, 1e-3),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'Do_error': PlotMeta(
        name='Random error in drizzle median diameter',
        cbar='RdYlGn_r',
        clabel=_DB,
        plot_range=(0.1, 0.5),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'mu': PlotMeta(
        name='Drizzle droplet size distribution shape parameter',
        cbar='viridis',
        clabel='',
        plot_range=(0, 10),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'S': PlotMeta(
        name='Backscatter-to-extinction ratio',
        cbar='viridis',
        clabel='',
        plot_range=(0, 25),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'S_error': PlotMeta(
        name='Random error in backscatter-to-extinction ratio',
        cbar='RdYlGn_r',
        clabel=_DB,
        plot_range=(0.1, 0.5),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'drizzle_N': PlotMeta(
        name='Drizzle number concentration',
        cbar='viridis',
        clabel=_M3,
        plot_range=(1e4, 1e9),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'drizzle_N_error': PlotMeta(
        name='Random error in drizzle number concentration',
        cbar='RdYlGn_r',
        clabel=_DB,
        plot_range=(0.1, 0.5),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'drizzle_lwc': PlotMeta(
        name='Drizzle liquid water content',
        cbar='viridis',
        clabel=_KGM3,
        plot_range=(1e-8, 1e-3),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'drizzle_lwc_error': PlotMeta(
        name='Random error in drizzle liquid water content',
        cbar='RdYlGn_r',
        clabel=_DB,
        plot_range=(0.3, 1),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'drizzle_lwf': PlotMeta(
        name='Drizzle liquid water flux',
        cbar='viridis',
        clabel=_KGM2S1,
        plot_range=(1e-8, 1e-5),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'drizzle_lwf_error': PlotMeta(
        name='Random error in drizzle liquid water flux',
        cbar='RdYlGn_r',
        clabel=_DB,
        plot_range=(0.3, 1),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'v_drizzle': PlotMeta(
        name='Drizzle droplet fall velocity',
        cbar='RdBu_r',
        clabel=_MS1,
        plot_range=(-2, 2),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'v_drizzle_error': PlotMeta(
        name='Random error in drizzle droplet fall velocity',
        cbar='RdYlGn_r',
        clabel=_DB,
        plot_range=(0.3, 1),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'drizzle_retrieval_status': PlotMeta(
        'Drizzle parameter retrieval status',
        clabel=_CLABEL['drizzle_retrieval_status'],
        plot_type='segment'
    ),
    'v_air': PlotMeta(
        name='Vertical air velocity',
        cbar='RdBu_r',
        clabel=_MS1,
        plot_range=(-2, 2),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'uwind': PlotMeta(
        name='Model zonal wind',
        cbar='RdBu_r',
        clabel=_MS1,
        plot_range=(-50, 50),
        plot_scale=_LIN,
        plot_type='model'
    ),
    'vwind': PlotMeta(
        name='Model meridional wind',
        cbar='RdBu_r',
        clabel=_MS1,
        plot_range=(-50, 50),
        plot_scale=_LIN,
        plot_type='model'
    ),
    'temperature': PlotMeta(
        name='Model temperature',
        cbar='RdBu_r',
        clabel='K',
        plot_range=(223.15, 323.15),
        plot_scale=_LIN,
        plot_type='model'
    ),
    'cloud_fraction': PlotMeta(
        name='Cloud fraction',
        cbar='Blues',
        clabel='',
        plot_range=(0, 1),
        plot_scale=_LIN,
        plot_type='model'
    ),
    'Tw': PlotMeta(
        name='Wet-bulb temperature',
        cbar='RdBu_r',
        clabel='K',
        plot_range=(223.15, 323.15),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'specific_humidity': PlotMeta(
        name='Model specific humidity',
        cbar='viridis',
        clabel='',
        plot_range=(1e-5, 1e-2),
        plot_scale=_LOG,
        plot_type='model'
    ),
    'q': PlotMeta(
        name='Model specific humidity',
        cbar='viridis',
        clabel='',
        plot_range=(1e-5, 1e-2),
        plot_scale=_LOG,
        plot_type='model'
    ),
    'pressure': PlotMeta(
        name='Model pressure',
        cbar='viridis',
        clabel='Pa',
        plot_range=(1e4, 1.5e5),
        plot_scale=_LOG,
        plot_type='model'
    ),
    'beta': PlotMeta(
        name='Attenuated backscatter coefficient',
        cbar='viridis',
        clabel=_SR1M1,
        plot_range=(1e-7, 1e-4),
        plot_scale=_LOG,
        plot_type='mesh'
     ),
    'beta_raw': PlotMeta(
        name='Raw attenuated backscatter coefficient',
        cbar='viridis',
        clabel=_SR1M1,
        plot_range=(1e-7, 1e-4),
        plot_scale=_LOG,
        plot_type='mesh'
     ),
    'beta_smooth': PlotMeta(
        name='Attenuated backscatter coefficient (smoothed)',
        cbar='viridis',
        clabel=_SR1M1,
        plot_range=(1e-7, 1e-3),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'Z': PlotMeta(
        name='Radar reflectivity factor',
        cbar='viridis',
        clabel=_DBZ,
        plot_range=(-40, 15),
        plot_scale=_LIN,     # already logarithmic
        plot_type='mesh'
     ),
    'Z_error': PlotMeta(
        name='Radar reflectivity factor random error',
        cbar='RdYlGn_r',
        clabel=_DB,
        plot_range=(0, 3),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'Ze': PlotMeta(
        name='Radar reflectivity factor',
        cbar='viridis',
        clabel=_DBZ,
        plot_range=(-40, 20),
        plot_scale=_LIN,     # already logarithmic
        plot_type='mesh'
     ),
    'ldr': PlotMeta(
        name='Linear depolarisation ratio',
        cbar='viridis',
        clabel=_DB,
        plot_range=(-30, -5),
        plot_scale=_LIN,     # already logarithmic
        plot_type='mesh'
     ),
    'width': PlotMeta(
        name='Spectral width',
        cbar='viridis',
        clabel=_MS1,
        plot_range=(1e-2, 1e0),
        plot_scale=_LOG,
        plot_type='mesh'
     ),
    'v': PlotMeta(
        name='Doppler velocity',
        cbar='RdBu_r',
        clabel=_MS1,
        plot_range=(-4, 4),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'v_sigma': PlotMeta(
        name='Standard deviation of mean velocity',
        cbar='viridis',
        clabel=_MS1,
        plot_range=(1e-2, 1e0),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'insect_prob': PlotMeta(
        name='Insect probability',
        cbar='viridius',
        clabel=_SR1M1,
        plot_range=(0, 1),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'radar_liquid_atten': PlotMeta(
        name='Approximate two-way radar attenuation due to liquid water',
        cbar='viridis',
        clabel=_DB,
        plot_range=(0, 5),
        plot_scale=_LIN,     # already logarithmic
        plot_type='mesh'
     ),
    'radar_gas_atten': PlotMeta(
        name='Two-way radar attenuation due to atmospheric gases',
        cbar='viridis',
        clabel=_DB,
        plot_range=(0, 1),
        plot_scale=_LIN,     # already logarithmic
        plot_type='mesh'
     ),
    'lwp': PlotMeta(
        name='Liquid water path',
        cbar='Blues',
        ylabel=_KGM2,
        plot_range=(0, 1),
        plot_scale=_LIN,
        plot_type='bar'
    ),
    'LWP': PlotMeta(
        name='Liquid water path',
        cbar='Blues',
        ylabel=_KGM2,
        plot_range=(0, 1),
        plot_scale=_LIN,
        plot_type='bar',
        source='mwr'
    ),
    'target_classification': PlotMeta(
        name='Target classification',
        clabel=_CLABEL['target_classification'],
        plot_type='segment'
    ),
    'detection_status': PlotMeta(
        name='Radar and lidar detection status',
        clabel=_CLABEL['detection_status'],
        plot_type='segment',
    ),
    'iwc': PlotMeta(
        name='Ice water content',
        cbar='viridis',
        clabel=_KGM3,
        plot_range=(1e-7, 1e-3),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'iwc_inc_rain': PlotMeta(
        name='Ice water content (including rain)',
        cbar='Blues',
        clabel=_KGM3,
        plot_range=(1e-7, 1e-4),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'iwc_error': PlotMeta(
        name='Ice water content error',
        cbar='RdYlGn_r',
        clabel=_DB,
        plot_range=(0, 5),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'iwc_retrieval_status': PlotMeta(
        name='Ice water content retrieval status',
        clabel=_CLABEL['iwc_retrieval_status'],
        plot_type='segment'
    ),
    'lwc': PlotMeta(
        name='Liquid water content',
        cbar='Blues',
        clabel=_KGM3,
        plot_range=(1e-5, 1e-2),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'lwc_error': PlotMeta(
        name='Liquid water content error',
        cbar='RdYlGn_r',
        clabel=_DB,
        plot_range=(0, 2),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'lwc_retrieval_status': PlotMeta(
        'Liquid water content retrieval status',
        clabel=_CLABEL['lwc_retrieval_status'],
        plot_type='segment'
    ),
    'droplet': PlotMeta(
        'Droplet bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    ),
    'falling': PlotMeta(
        'Falling bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    ),
    'cold': PlotMeta(
        'Cold bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    ),
    'melting': PlotMeta(
        'Melting bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    ),
    'aerosol': PlotMeta(
        'Aerosol bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    ),
    'insect': PlotMeta(
        'Insect bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    ),
    'radar': PlotMeta(
        'Radar bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    ),
    'lidar': PlotMeta(
        'Lidar bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    ),
    'clutter': PlotMeta(
        'Clutter bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    ),
    'molecular': PlotMeta(
        'Molecular bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    ),
    'attenuated': PlotMeta(
        'Attenuated bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    ),
    'corrected': PlotMeta(
        'Corrected bit',
        cbar=_CBAR['bit'],
        plot_range=(0, 1),
        plot_type='bit'
    )
}
