# Tähän tiedostoon kirjataan kaikkien tuotteiden tarvittavat tiedot, jotta saadaan tietyn tyyppiset plotit tehtyä
from collections import namedtuple

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

    'iwc':'jet',
    'iwc_error':'jet',

    'iwc_retrieval_status':
        ("#f8f8ff", "#00bfff", "#ff4500", "#0000ff", "#ffff00",
         "#2f4f4f", "#778899", "#d3d3d3"),

    'iwc_sensitivity': 'jet',
    'iwc_bias': 'jet',
    'iwc_inc_rain':'jet',

    'lwc':'jet',
    'lwc_error':'jet',
    'lwc_th':'jet',
    
    'lwc_retrieval_status':
        ("#f8f8ff", "#00bfff", "#0000ff", "#2f4f4f", "#ffa500", 
         "#ff4500", "#2f4f4f"),
    
    'lwp':'jet',
    'lwp_error':'jet'
}

ATTRIBUTES = {
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
        cbar=_CBAR['iwc'],
        clabel='$kg$'+' $m^{-3}$',
        plot_range=(1e-7, 1e-3),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'iwc_error': PlotMeta(
        'Random error in ice water content, one standard deviation',
        cbar=_CBAR['iwc_error'],
        clabel='dB',
        plot_range=(0, 3),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'iwc_bias': PlotMeta(
        'Possible bias in ice water content, one standard deviation',
        clabel='dB',
        plot_type='mesh'
    ),
    'iwc_sensitivity': PlotMeta(
        'Minimum detectable ice water content',
        clabel='$kg$'+' $m^{-3}$',
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
        cbar=_CBAR['iwc_inc_rain'],
        clabel='$kg$'+' $m^{-3}$',
        plot_range=(1e-5, 1e-2),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'lwc': PlotMeta(
        'Liquid water content',
        cbar=_CBAR['lwc'],
        clabel='$kg$'+' $m^{-3}$',
        plot_range=(1e-5, 1e-2),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'lwc_error': PlotMeta(
        'Random error in liquid water content, one standard deviation',
        clabel='dB',
        plot_range=(0,2),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'lwc_retrieval_status': PlotMeta(
        'Liquid water content retrieval status',
        cbar=_CBAR['lwc_retrieval_status'],
        clabel=_CLABEL['lwc_retrieval_status'],
        plot_type='segment'
    ),
    'lwp': PlotMeta(
        'Liquid water path',
        cbar=_CBAR['lwp'],
        clabel='$kg$'+' $m^{-2}$',
        plot_range=(-100, 1000),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'lwp_error': PlotMeta(
        'Random error in liquid water path, one standard deviation',
        cbar=_CBAR['lwp_error'],
        clabel='$kg$'+' $m^{-2}$',
        plot_type='mesh'
    ),
    'lwc_th': PlotMeta(
        'Liquid water content (tophat distribution)',
        cbar=_CBAR['lwc_th'],
        clabel='$kg$'+' $m^{-3}$',
        plot_range=(1e-5, 1e-2),
        plot_scale=_LOG,
        plot_type='mesh'
    )
}