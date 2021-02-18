"""CloudnetArray class."""
import math
from typing import Optional, Union
import numpy as np
import numpy.ma as ma
import netCDF4
from cloudnetpy import utils
from cloudnetpy.metadata import MetaData


class CloudnetArray:
    """Stores netCDF4 variables, numpy arrays and scalars as CloudnetArrays.

    Args:
        variable: The netCDF4 :class:`Variable` instance, numpy array (masked or regular),
            or scalar (float, int).
        name: Name of the variable.
        units_from_user: Units of the variable.

    Attributes:
        name (str): Name of the variable.
        data (ndarray): The actual data.
        data_type (str): 'i4' for integers, 'f4' for floats.
        units (str): The `units_from_user` argument if it is given. Otherwise
            copied from the original netcdf4 variable. Empty if input is just data.

    """

    def __init__(self,
                 variable: Union[netCDF4.Variable, np.ndarray, float, int],
                 name: str,
                 units_from_user: Optional[str] = None):
        self.variable = variable
        self.name = name
        self.data = self._init_data()
        self.units = self._init_units(units_from_user)
        self.data_type = self._init_data_type()

    def lin2db(self) -> None:
        """Converts linear units to log."""
        if 'db' not in self.units.lower():
            self.data = utils.lin2db(self.data)
            self.units = 'dB'

    def db2lin(self) -> None:
        """Converts log units to linear."""
        if 'db' in self.units.lower():
            self.data = utils.db2lin(self.data)
            self.units = ''

    def mask_indices(self, ind: list) -> None:
        """Masks data from given indices."""
        self.data[ind] = ma.masked

    def rebin_data(self,
                   time: np.ndarray,
                   time_new: np.ndarray,
                   height: Optional[np.ndarray] = None,
                   height_new: Optional[np.ndarray] = None) -> None:
        """Rebins `data` in time and optionally interpolates in height.

        Args:
            time: 1D time array.
            time_new: 1D new time array.
            height: 1D height array.
            height_new: 1D new height array. Should be given if also `height` is given.

        """
        if self.data.ndim == 1:
            self.data = utils.rebin_1d(time, self.data.astype(float), time_new)
        else:
            self.data = utils.rebin_2d(time, self.data, time_new)
            if np.any(height) and np.any(height_new):
                self.data = utils.interpolate_2d_masked(self.data,
                                                        (time_new, height),
                                                        (time_new, height_new))

    def fetch_attributes(self) -> list:
        """Returns list of user-defined attributes."""
        attributes = []
        for attr in self.__dict__:
            if attr not in ('name', 'data', 'data_type', 'variable'):
                attributes.append(attr)
        return attributes

    def set_attributes(self, attributes: MetaData) -> None:
        """Overwrites existing instance attributes."""
        for key in attributes._fields:  # To iterate namedtuple fields.
            data = getattr(attributes, key)
            if data:
                setattr(self, key, data)

    def _init_data(self) -> np.ndarray:
        if isinstance(self.variable, netCDF4.Variable):
            return self.variable[:]
        if isinstance(self.variable, np.ndarray):
            return self.variable
        if isinstance(self.variable, (int, float)):
            return np.array(self.variable)
        if isinstance(self.variable, str):
            try:
                numeric_value = utils.str_to_numeric(self.variable)
                return np.array(numeric_value)
            except ValueError:
                pass
        raise ValueError(f'Incorrect CloudnetArray input: {self.variable}')

    def _init_units(self, units_from_user: Union[str, None]) -> str:
        if units_from_user is not None:
            return units_from_user
        return getattr(self.variable, 'units', '')

    def _init_data_type(self) -> str:
        if self.data.dtype in (np.float32, np.float64):
            return 'f4'
        return 'i4'

    def __getitem__(self, ind: tuple) -> np.ndarray:
        return self.data[ind]


class RadarArray(CloudnetArray):
    """The :class:`RadarArray` class, child of :class:`CloudnetArray`.

    This class contains additional, cloud radar -specific methods.

    """

    def filter_isolated_pixels(self) -> None:
        """Filter vertical artifacts in radar data.

        Notes:
            These kind of artifacts are seen in RPG data.

        """
        is_data = (~self.data.mask).astype(int)
        is_data_filtered = utils.filter_x_pixels(is_data)
        self.data[is_data_filtered == 0] = ma.masked

    def calc_linear_std(self, time: np.ndarray, time_new: np.ndarray) -> None:
        """Calculates std of radar velocity.

        Args:
            time: 1D time array.
            time_new: 1D new time array.

        Notes:
            The result is masked if the bin contains masked values.

        """
        self.data = utils.rebin_2d(time, self.data.astype(float), time_new, 'std')

    def rebin_velocity(self,
                       time: np.ndarray,
                       time_new: np.ndarray,
                       folding_velocity: Union[float, list],
                       sequence_indices: list) -> None:
        """Rebins Doppler velocity in polar coordinates.

        Args:
            time: 1D time array.
            time_new: 1D new time array.
            folding_velocity: Folding velocity (m/s). Can be float when it's the same for all
                altitudes, or list when it matches difference altitude regions
                (defined in `sequence_indices`).
            sequence_indices: List containing indices of different folding regions,
                e.g. [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10]].

        """
        def _get_scaled_vfold() -> Union[float, list]:
            vfold_scaled = math.pi / folding_velocity
            if isinstance(vfold_scaled, float):
                vfold_scaled = [vfold_scaled]
            return vfold_scaled

        def _scale_by_vfold(data_in: np.array, fun) -> np.array:
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
