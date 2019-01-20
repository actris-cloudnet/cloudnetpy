"""CloudnetArray class."""
import math
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils


class CloudnetArray():
    """Transforms Cloudnet variables into NetCDF variables.

    Attributes:
        name (str): Name of the variable.
        data (array_like): The actual data.
        data_type (str): 'i4' for integers, 'f4' for floats.
        units (str): Copied from the original netcdf4
            variable (if existing).

    """

    def __init__(self, netcdf4_variable, name, units=None):
        self.name = name
        self.data = self._get_data(netcdf4_variable)
        self.data_type = self._init_data_type()
        self.units = self._init_units(units, netcdf4_variable)

    def __getitem__(self, ind):
        return self.data[ind]

    @staticmethod
    def _get_data(array):
        return array if utils.isscalar(array) else array[:]

    @staticmethod
    def _init_units(units_from_user, netcdf4_variable):
        if units_from_user:
            return units_from_user
        elif hasattr(netcdf4_variable, 'units'):
            return netcdf4_variable.units
        return ''

    def _init_data_type(self):
        if ((isinstance(self.data, np.ndarray) and self.data.dtype
             in (np.float32, np.float64)) or isinstance(self.data, float)):
            return 'f4'
        return 'i4'

    def lin2db(self):
        """Converts linear units do log."""
        if 'db' not in self.units.lower():
            self.data = utils.lin2db(self.data)
            self.units = 'dB'

    def db2lin(self):
        """Converts log units to linear."""
        if 'db' in self.units.lower():
            self.data = utils.db2lin(self.data)
            self.units = ''

    def rebin_data(self, time, time_new, height=None, height_new=None):
        """Rebins data in time and optionally in height."""
        self.data = utils.rebin_2d(time, self.data, time_new)
        if np.any(height) and np.any(height_new):
            self.data = utils.rebin_2d(height, self.data.T, height_new).T

    def rebin_in_polar(self, time, time_new, folding_velocity):
        """Rebins velocity in polar coordinates."""
        folding_velocity_scaled = math.pi / folding_velocity
        data_scaled = self.data * folding_velocity_scaled
        vel_x, vel_y = np.cos(data_scaled), np.sin(data_scaled)
        vel_x_mean = utils.rebin_2d(time, vel_x, time_new)
        vel_y_mean = utils.rebin_2d(time, vel_y, time_new)
        self.data = np.arctan2(vel_y_mean, vel_x_mean) / folding_velocity_scaled

    def mask_indices(self, ind):
        """Masks data from given indices."""
        self.data[ind] = ma.masked

    def fetch_attributes(self):
        """Returns list of user-defined attributes."""
        for attr in self.__dict__:
            if attr not in ('name', 'data', 'data_type'):
                yield attr

    def set_attributes(self, attributes):
        """Set some attributes if they exist."""
        for key in attributes._fields:
            data = getattr(attributes, key)
            if data:
                setattr(self, key, data)
