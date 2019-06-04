"""Metadata for plotting module."""
from collections import namedtuple

FIELDS = ('name',
          'cbar',
          'clabel',
          'ylabel',
          'plot_range',
          'plot_scale',
          'plot_type')

PlotMeta = namedtuple('PlotMeta', FIELDS, defaults=(None,)*len(FIELDS))

_LOG = 'logarithmic'
_LIN = 'linear'

_M3 = '$m^{-3}$'
_MS1 = 'm s$^{-1}$'
_SR1M1 = 'sr$^{-1}$ m$^{-1}$'
_KGM2 = 'kg m$^{-2}$'
_KGM3 = 'kg m$^{-3}$'
_KGM2S1 = 'kg m$^{-2} s$^{-1}$'
_DB = 'dB'
_DBZ = 'dBZ'

_CLABEL = {
    'target_classification':
        ("Clear sky",
         "Cloud droplets only",
         "Drizzle or rain",
         "Drizzle/rain or cloud droplets",
         "Ice",
         "Ice & supercooled droplets",
         "Melting ice",
         "Melting ice & cloud droplets",
         "Aerosols",
         "Insects",
         "Aerosols & insects"),

    'detection_status':
        ("Clear sky",
         "Lidar echo only",
         "Radar echo but uncorrected atten.",
         "Good radar & lidar echos",
         "No radar but unknown atten.",
         "Good radar echo only",
         "No radar but known atten.",
         "Radar corrected for liquid atten.",
         "Radar ground clutter",
         "Lidar molecular scattering"),

    'iwc_retrieval_status':
        ("No ice",
         "Reliable retrieval",
         "Unreliable: uncorrected attenuation",
         "Retrieval with correction for liquid atten.",
         "Ice detected only by the lidar",
         "Ice above rain: no retrieval",
         "Clear sky above rain",
         "Would be identified as ice if below freezing"),

    'lwc_retrieval_status':
        ("No liquid water",
         "Reliable retrieval",
         "Adiabatic retrieval: cloud top adjusted",
         "Adiabatic retrieval: new cloud pixel",
         "Unreliable lwp: no retrieval",
         "Unreliable lwp/cloud boundaries: no retrieval",
         "Rain present: no retrieval")
}

_CBAR = {
    'target_classification':
        ("#ffffff", "#00bfff", "#ff4500", "#0000ff", "#ffff00", "#32cd32",
         "#ffa500", "#66cdaa", "#d3d3d3", "#778899", "#2f4f4f"),
    'detection_status':
        ("#ffffff", "#ffff00", "#2f4f4f", "#3cb371", "#778899", "#87cefa",
         "#d3d3d3", "#0000ff", "#ff4500", "#ffa500"),
    'iwc_retrieval_status':
        ("#ffffff", "#00bfff", "#ff4500", "#0000ff", "#ffff00",
         "#2f4f4f", "#778899", "#d3d3d3"),
    'lwc_retrieval_status':
        ("#ffffff", "#00bfff", "#0000ff", "#3cb371", "#ffa500",
         "#ff4500", "#ECECEC"),
    'bit':
        ("#ffffff", "#4682b4")
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
    'drizzle_N': PlotMeta(
        name='Drizzle number concentration',
        cbar='viridis',
        clabel=_M3,
        plot_range=(1e4, 1e9),
        plot_scale=_LOG,
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
    'v_drizzle': PlotMeta(
        name='Drizzle droplet fall velocity',
        cbar='RdBu_r',
        clabel=_MS1,
        plot_range=(-2, 2),
        plot_scale=_LIN,
        plot_type='mesh'
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
        plot_range=(203, 303),
        plot_scale=_LIN,
        plot_type='model'
    ),
    'Tw': PlotMeta(
        name='Wet-bulb temperature',
        cbar='RdBu_r',
        clabel='K',
        plot_range=(203, 303),
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
        plot_range=(1e-7, 1e-2),
        plot_scale=_LOG,
        plot_type='mesh'
     ),
    'beta_raw': PlotMeta(
        name='Raw attenuated backscatter coefficient',
        cbar='viridis',
        clabel=_SR1M1,
        plot_range=(1e-7, 1e-3),
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
        plot_range=(-35, -10),
        plot_scale=_LIN,     # already logarithmic
        plot_type='mesh'
     ),
    'width': PlotMeta(
        name='Spectral width',
        cbar='viridis',
        clabel=_MS1,
        plot_range=(1e-1, 1e0),
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
        name='STD of Doppler velocity',
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
        plot_range=(0, 10),
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
    'target_classification': PlotMeta(
        name='Target classification',
        cbar=_CBAR['target_classification'],
        clabel=_CLABEL['target_classification'],
        plot_type='segment'
    ),
    'detection_status': PlotMeta(
        name='Radar and lidar detection status',
        cbar=_CBAR['detection_status'],
        clabel=_CLABEL['detection_status'],
        plot_type='segment'
    ),
    'iwc': PlotMeta(
        name='Ice water content',
        cbar='Blues',
        clabel=_KGM3,
        plot_range=(1e-7, 1e-4),
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
        cbar=_CBAR['iwc_retrieval_status'],
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
        cbar=_CBAR['lwc_retrieval_status'],
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
