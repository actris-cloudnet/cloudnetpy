"""
Metadata for old Cloudnet files, enabling us to
plot the old files and compare with the new files.
"""
import numpy as np
import numpy.ma as ma
from collections import namedtuple

FIELDS = ('hidden',
          'swapped',
          'rename')

PlotMeta = namedtuple('PlotMeta', FIELDS)
PlotMeta.__new__.__defaults__ = (None,) * len(FIELDS)

_HIDDEN = {
    'detection_status':
        ("Clear sky",
         "Lidar echo only",
         "Radar corrected for liquid atten",
         "Good radar & lidar echos",
         "_No radar but unknown atten.",
         "Good radar echo only",
         "_No radar mut known atten.",
         "Radar uncorrected for liquid atten.",
         "Radar ground clutter",
         "Lidar molecular scattering"),

    'lwc_retrieval_status':
        ("No liquid water",
         "Reliable retrieval",
         "Adiabatic retrieval: cloud top adjusted",
         "Adiabatic retrieval: new cloud pixel",
         "_Unreliable lwp",
         "_Unreliable lwp/cloud boundaries",
         "Rain present: no retrieval")
}

ATTRIBUTES = {
    'specific_humidity': PlotMeta(
        rename='q'
    ),
    'detection_status': PlotMeta(
        hidden=_HIDDEN['detection_status'],
        swapped=[(1, 4), (2, 5), (1, 3), (2, 3), (3, 4)]
    ),
    'iwc_retrieval_status': PlotMeta(
        swapped=[(2, 3)]
    ),
    'lwc_retrieval_status': PlotMeta(
        hidden=_HIDDEN['lwc_retrieval_status']
    )
}


def fix_legacy_data(data, name):
    def _remove_data():
        segments = [x[0] for x in ATTRIBUTES[name].hidden]
        segments_to_hide = np.char.startswith(segments, '_')
        indices = np.where(segments_to_hide)[0]
        for ind in np.flip(indices):
            data[data == ind] = ma.masked
            data[data > ind] -= 1

    def _swap_data():
        def _swap(arr):
            ind_a = np.where(data == a)
            ind_b = np.where(data == b)
            arr[ind_a], arr[ind_b] = b, a

        for a, b in ATTRIBUTES[name].swapped:
            _swap(data)

    if name in ATTRIBUTES.keys():
        if ATTRIBUTES[name].hidden:
            _remove_data()
        if ATTRIBUTES[name].swapped:
            _swap_data()
        if ATTRIBUTES[name].rename:
            name = ATTRIBUTES[name].rename

    return data, name
