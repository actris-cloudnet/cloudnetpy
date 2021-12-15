"""
Metadata for old Cloudnet files, enabling us to
plot the old files and compare with the new files.
"""
from collections import namedtuple
import numpy as np
import numpy.ma as ma

FIELDS = ('hidden',
          'swapped',
          'rename')

PlotMeta = namedtuple('PlotMeta', FIELDS)
PlotMeta.__new__.__defaults__ = (None,) * len(FIELDS)


ATTRIBUTES = {
    'specific_humidity': PlotMeta(
        rename='q'
    ),
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
