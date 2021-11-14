"""Module for reading raw cloud radar data."""
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils
from cloudnetpy.categorize import DataSource


class NcRadar(DataSource):
    """Class for radars providing netCDF files. Child of DataSource().

    Args:
        full_path: Filename of a radar-produced netCDF file.
        site_meta: Some metadata of the site.

    Notes:
        Used with BASTA and MIRA radars.
    """
    def __init__(self, full_path: str, site_meta: dict):
        super().__init__(full_path)
        self.site_meta = site_meta
        self.date = None
        self.instrument = None

    def init_data(self, keymap: dict) -> None:
        """Reads selected fields and fixes the names."""
        for key in keymap:
            name = keymap[key]
            array = self.getvar(key)
            array = np.array(array) if utils.isscalar(array) else array
            array[~np.isfinite(array)] = ma.masked
            self.append_data(array, name)

    def add_time_and_range(self) -> None:
        """Adds time and range."""
        range_instru = np.array(self.getvar('range'))
        time = np.array(self.time)
        self.append_data(range_instru, 'range')
        self.append_data(time, 'time')
