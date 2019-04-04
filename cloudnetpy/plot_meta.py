"""Metadata for plotting module."""
from collections import namedtuple

"""Information for plotting of all parameter in CloudnetPy"""

#TODO: - Lisää tarvittaviin muuttujiin yksikkömuutos termi, että plottaus aina
#        samassa yksikössä. Oletus, että alkuperäisyksikkö aina sama.
#      - Muokkaa colorbaria jetissä siten, että minimi valkoinen ja max musta

FIELDS = ('name',
          'cbar',
          'clabel',
          'xlabel',
          'ylabel',
          'plot_range',
          'plot_scale',
          'plot_type')

PlotMeta = namedtuple('PlotMeta', FIELDS, defaults=(None,)*len(FIELDS))

_LOG = 'logarithmic'
_LIN = 'linear'
_KGM2 = '$kg$'+' $m^{-2}$'
_KGM3 = '$kg$'+' $m^{-3}$'
_MS2 = '$m$' + ' $s^{-1}$'

_CLABEL = {
    'target_classification':
        ("Clear sky",
         "Cloud droplets only",
         "Drizzle or rain",
         "Drizzle/rain or cloud droplets",
         "Ice", "Ice & supercooled droplets",
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
         "Adiabatic ret.: cloud top adjusted",
         "Adiabatic ret.: new cloud pixel",
         "Unreliable lwp: no ret.",
         "Unreliable lwp/cloud boundaries: no ret.",
         "Rain present: no ret.")
}

_CBAR = {
    'target_classification':
        ("#f8f8ff", "#00bfff", "#ff4500", "#0000ff", "#ffff00", "#32cd32",
         "#ffa500", "#66cdaa", "#d3d3d3", "#778899", "#2f4f4f"),
    'detection_status':
        ("#f8f8ff", "#ffff00", "#2f4f4f", "#3cb371", "#778899", "#87cefa",
         "#d3d3d3", "#0000ff", "#ff4500", "#ffa500"),
    'iwc_retrieval_status':
        ("#f8f8ff", "#00bfff", "#ff4500", "#0000ff", "#ffff00",
         "#2f4f4f", "#778899", "#d3d3d3"),
    'lwc_retrieval_status':
        ("#f8f8ff", "#00bfff", "#0000ff", "#2f4f4f", "#ffa500",
         "#ff4500", "#2f4f4f"),
}

ATTRIBUTES = {
    # Categorize
    'rain_rate': PlotMeta(
        'Rain rate',
        cbar='jet',
        clabel='$mm$' + ' $h^{-1}$',
        plot_range=(0, 50),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'beta': PlotMeta(
        'Attenuated backscatter coefficient',
        cbar='jet',
        clabel='$sr^{-1}$' + ' $m^{-1}$',
        plot_range=(1e-7, 1e-3),
        plot_scale=_LOG,
        plot_type='mesh'
     ),
    'beta_raw': PlotMeta(
        'Attenuated backscatter coefficient',
        cbar='jet',
        clabel='$sr^{-1}$' + ' $m^{-1}$',
        plot_range=(1e-7, 1e-3), #*10?
        plot_scale=_LOG,
        plot_type='mesh'
     ),
    'Z': PlotMeta(
        'Radar reflectivity factor',
        cbar='jet',
        clabel='$dBZ$',
        plot_range=(-40, 30),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'Ze': PlotMeta(
        'Radar reflectivity factor',
        cbar='jet',
        clabel='$dBZ$',
        plot_range=(-40, 20),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'ldr': PlotMeta(
        'Linear depolarisation ratio',
        cbar='viridius',
        clabel='$dB$',
        plot_range=(-35, -10),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'width': PlotMeta(
        'Spectral width',
        cbar='viridius',
        clabel=_MS2,
        plot_range=(0, 1),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'v': PlotMeta(
        'Doppler velocity',
        cbar='RdBu_r',
        clabel=_MS2,
        plot_range=(-4, 2),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'insect_prob': PlotMeta(
        'Attenuated backscatter coefficient',
        cbar='viridius',
        clabel='$sr^{-1}$' + ' $m^{-1}$',
        plot_range=(0, 1),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'radar_liquid_atten': PlotMeta(
        'Approximate two-way radar attenuation due to liquid water',
        cbar='viridius',
        clabel='$dB$',
        plot_range=(0, 10),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'radar_gas_atten': PlotMeta(
        'Two-way radar attenuation due to atmospheric gases',
        cbar='viridius',
        clabel='$dB$',
        plot_range=(0, 1),
        plot_scale=_LIN,
        plot_type='mesh'
     ),
    'lwp': PlotMeta(
        'Liquid water path',
        cbar='Blues',
        clabel=_KGM2,
        plot_range=(-100, 1000),
        plot_scale=_LIN,
        plot_type='bar'
    ),

    # products
    'target_classification': PlotMeta(
        'Target classification',
        cbar=_CBAR['target_classification'],
        clabel=_CLABEL['target_classification'],
        plot_type='segment'
    ),
    'detection_status': PlotMeta(
        'Radar and lidar detection status',
        cbar=_CBAR['detection_status'],
        clabel=_CLABEL['detection_status'],
        plot_type='segment'
    ),
    'iwc': PlotMeta(
        'Ice water content',
        cbar='jet',
        clabel=_KGM3,
        plot_range=(1e-7, 1e-3),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'iwc_error': PlotMeta(
        'Random error in ice water content, one standard deviation',
        cbar='jet',
        clabel='dB',
        plot_range=(0, 3),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'iwc_retrieval_status': PlotMeta(
        'Ice water content retrieval status',
        cbar=_CBAR['iwc_retrieval_status'],
        clabel=_CLABEL['iwc_retrieval_status'],
        plot_type='segment'
    ),
    'iwc_inc_rain': PlotMeta(
        'Ice water content including rain',
        cbar='jet',
        clabel=_KGM3,
        plot_range=(1e-5, 1e-2),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'lwc': PlotMeta(
        'Liquid water content',
        cbar='jet',
        clabel=_KGM3,
        plot_range=(1e-5, 1e-2),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'lwc_error': PlotMeta(
        'Random error in liquid water content, one standard deviation',
        clabel='dB',
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
    'lwc_th': PlotMeta(
        'Liquid water content (tophat distribution)',
        cbar='jet',
        clabel=_KGM3,
        plot_range=(1e-5, 1e-2),
        plot_scale=_LOG,
        plot_type='mesh'
    )
}