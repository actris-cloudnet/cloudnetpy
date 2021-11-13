"""Module for reading raw cloud radar data."""
import logging
import numpy as np
import numpy.ma as ma
from cloudnetpy import RadarArray, CloudnetArray, utils
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
        super().__init__(full_path, radar=True)
        self.site_meta = site_meta
        self.range = self.getvar(self, 'range')
        self.date = None
        self.instrument = None

    def init_data(self, keymap: dict) -> None:
        """Reads correct fields and fixes the names."""
        for key in keymap:
            name = keymap[key]
            array = self.getvar(key)
            array = np.array(array) if utils.isscalar(array) else array
            array[~np.isfinite(array)] = ma.masked
            self.append_data(array, name)

    def linear_to_db(self, variables_to_log: tuple) -> None:
        """Changes linear units to logarithmic."""
        for name in variables_to_log:
            self.data[name].lin2db()

    def add_meta(self) -> None:
        """Adds metadata."""
        self.append_data(self.instrument.frequency, 'radar_frequency')
        for key in ('time', 'range'):
            self.append_data(np.array(getattr(self, key)), key)
        possible_nyquist_names = ('ambiguous_velocity', 'NyquistVelocity')
        self._unknown_variable_to_cloudnet_array(possible_nyquist_names, 'nyquist_velocity',
                                                 ignore_mask=True)

    def add_height(self):
        if 'altitude' in self.data:
            try:
                elevation = self.getvar('elv', 'elevation')
                tilt_angle = 90 - ma.median(elevation)
            except RuntimeError:
                logging.warning('Assuming 90 deg elevation')
                tilt_angle = 0
            height = utils.range_to_height(self.getvar('range'), tilt_angle)
            height += float(self.data['altitude'].data)
            height = np.array(height)
            self.data['height'] = CloudnetArray(height, 'height')

    def add_site_meta(self) -> None:
        for key, value in self.site_meta.items():
            if key in ('latitude', 'longitude', 'altitude'):
                self.append_data(float(value), key)
