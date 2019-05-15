"""CloudnetArray class."""
import math
import numpy as np
import numpy.ma as ma
from . import utils


class CloudnetArray:
    """Stores NetCDF variables as CloudnetArrays.

    Attributes:
        name (str): Name of the variable.
        data (array_like): The actual data.
        data_type (str): 'i4' for integers, 'f4' for floats.
        units (str): Copied from the original netcdf4
            variable (if existing).

    """

    def __init__(self, netcdf4_variable, name, units_from_user=None):
        self.name = name
        self.data = self._get_data(netcdf4_variable)
        self.data_type = self._init_data_type()
        self.units = self._init_units(units_from_user, netcdf4_variable)

    def __getitem__(self, ind):
        return self.data[ind]

    @staticmethod
    def _get_data(array):
        return array if utils.isscalar(array) else array[:]

    @staticmethod
    def _init_units(units_from_user, netcdf4_variable):
        if units_from_user:
            return units_from_user
        return getattr(netcdf4_variable, 'units', '')

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

    def calc_linear_std(self, time, time_new):
        """Calculates std of velocity.

        The result is masked if the bin contains masked values.

        """
        self.data = utils.rebin_2d(time, self.data.astype(float), time_new, 'std')

    def rebin_1d_data(self, time, time_new):
        """Rebins 1D array in time."""
        self.data = utils.rebin_1d(time, self.data.astype(float), time_new)

    def rebin_in_polar(self, time, time_new, folding_velocity,
                       sequence_indices):
        """Rebins velocity in polar coordinates.

        Velocity needs to be averaged in polar coordinates due to folding.

        """
        def _scale(source, target, fun):
            for i, ind in enumerate(sequence_indices):
                target[:, ind] = fun(source[:, ind], folding_velocity_scaled[i])

        def _get_scaled_vfold():
            vfold_scaled = math.pi / folding_velocity
            if isinstance(vfold_scaled, float):
                vfold_scaled = [vfold_scaled]
            return vfold_scaled

        folding_velocity_scaled = _get_scaled_vfold()
        data_scaled = ma.copy(self.data)
        _scale(self.data, data_scaled, np.multiply)
        vel_x = np.cos(data_scaled)
        vel_y = np.sin(data_scaled)
        vel_x_mean = utils.rebin_2d(time, vel_x, time_new)
        vel_y_mean = utils.rebin_2d(time, vel_y, time_new)
        mean_vel_scaled = np.arctan2(vel_y_mean, vel_x_mean)
        self.data = self.data[:len(time_new), :]
        _scale(mean_vel_scaled, self.data, np.divide)

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
