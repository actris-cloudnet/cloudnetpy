"""CloudnetArray class."""
import math
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils


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
        """Converts linear units to log."""
        if 'db' not in self.units.lower():
            self.data = utils.lin2db(self.data)
            self.units = 'dB'

    def db2lin(self):
        """Converts log units to linear."""
        if 'db' in self.units.lower():
            self.data = utils.db2lin(self.data)
            self.units = ''

    def rebin_data(self, time, time_new, height=None, height_new=None):
        """Rebins data in time and optionally interpolates in height."""
        self.data = utils.rebin_2d(time, self.data, time_new)
        if np.any(height) and np.any(height_new):
            self.data = utils.interpolate_2d_masked(self.data,
                                                    (time_new, height),
                                                    (time_new, height_new))

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
        def _get_scaled_vfold():
            vfold_scaled = math.pi / folding_velocity
            if isinstance(vfold_scaled, float):
                vfold_scaled = [vfold_scaled]
            return vfold_scaled

        def _scale_by_vfold(data_in, fun):
            data_out = ma.copy(data_in)
            for i, ind in enumerate(sequence_indices):
                data_out[:, ind] = fun(data_in[:, ind], folding_velocity_scaled[i])
            return data_out

        folding_velocity_scaled = _get_scaled_vfold()
        data_scaled = _scale_by_vfold(self.data, np.multiply)
        vel_x = np.cos(data_scaled)
        vel_y = np.sin(data_scaled)
        vel_x_mean = utils.rebin_2d(time, vel_x, time_new)
        vel_y_mean = utils.rebin_2d(time, vel_y, time_new)
        mean_vel_scaled = np.arctan2(vel_y_mean, vel_x_mean)
        self.data = _scale_by_vfold(mean_vel_scaled, np.divide)

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

    def filter_isolated_pixels(self):
        """Filter vertical artifacts in cloud radar data."""
        is_data = (~self.data.mask).astype(int)
        is_data_filtered = utils.filter_x_pixels(is_data)
        self.data[is_data_filtered == 0] = ma.masked
