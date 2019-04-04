"""Metadata for plotting module."""
from collections import namedtuple

FIELDS = ('name',
          'cbar',
          'clabel',
          'plot_range',
          'plot_scale',
          'plot_type')

PlotMeta = namedtuple('PlotMeta', FIELDS, defaults=(None,)*len(FIELDS))

_LOG = 'logarithmic'
_LIN = 'linear'
_KGM2 = '$kg$'+' $m^{-2}$'
_KGM3 = '$kg$'+' $m^{-3}$'

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
        cbar='jet',
        clabel=_KGM3,
        plot_range=(1e-7, 1e-3),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'iwc_inc_rain': PlotMeta(
        name='Ice water content including rain',
        cbar='jet',
        clabel=_KGM3,
        plot_range=(1e-7, 1e-4),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'iwc_error': PlotMeta(
        name='Random error in ice water content, one standard deviation',
        cbar='jet',
        clabel='dB',
        plot_range=(0, 3),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'iwc_retrieval_status': PlotMeta(
        name='Ice water content retrieval status',
        cbar=_CBAR['iwc_retrieval_status'],
        clabel=_CLABEL['iwc_retrieval_status'],
        plot_type='segment'
    ),
    'iwc_bias': PlotMeta(
        name='Possible bias in ice water content, one standard deviation',
        clabel='dB',
    ),
    'iwc_sensitivity': PlotMeta(
        name='Minimum detectable ice water content',
        clabel=_KGM3,
    ),
    'lwc': PlotMeta(
        name='Liquid water content',
        cbar='jet',
        clabel=_KGM3,
        plot_range=(1e-5, 1e-2),
        plot_scale=_LOG,
        plot_type='mesh'
    ),
    'lwc_error': PlotMeta(
        name='Random error in liquid water content, one standard deviation',
        cbar='jet',
        clabel='dB',
        plot_range=(0, 2),
        plot_scale=_LIN,
        plot_type='mesh'
    ),
    'lwc_retrieval_status': PlotMeta(
        name='Liquid water content retrieval status',
        cbar=_CBAR['lwc_retrieval_status'],
        clabel=_CLABEL['lwc_retrieval_status'],
        plot_type='segment'
    ),
    'lwp': PlotMeta(
        name='Liquid water path',
        clabel=_KGM2,
        plot_range=(0, 1),
        plot_scale=_LIN,
    ),
    'lwp_error': PlotMeta(
        name='Random error in liquid water path, one standard deviation',
        clabel=_KGM2,
    ),
}
