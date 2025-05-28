"""CloudnetArray class."""

from collections.abc import Sequence

import netCDF4
import numpy as np
from numpy import ma
from scipy.interpolate import interp1d

from cloudnetpy import utils
from cloudnetpy.metadata import MetaData


class CloudnetArray:
    """Stores netCDF4 variables, numpy arrays and scalars as CloudnetArrays.

    Args:
        variable: The netCDF4 :class:`Variable` instance,
            numpy array (masked or regular), or scalar (float, int).
        name: Name of the variable.
        units_from_user: Explicit units, optional.
        dimensions: Explicit dimension names, optional.
        data_type: Explicit data type, optional.
        source: Source attribute, optional.

    """

    def __init__(
        self,
        variable: netCDF4.Variable | np.ndarray | float,
        name: str,
        units_from_user: str | None = None,
        dimensions: Sequence[str] | None = None,
        data_type: str | None = None,
        source: str | None = None,
    ):
        self.variable = variable
        self.name = name
        self.data = self._init_data()
        self.units = units_from_user or self._init_units()
        self.data_type = data_type or self._init_data_type()
        self.dimensions = dimensions
        self.source = source

    def lin2db(self) -> None:
        """Converts linear units to log."""
        if "db" not in self.units.lower():
            self.data = utils.lin2db(self.data)
            self.units = "dB"

    def db2lin(self) -> None:
        """Converts log units to linear."""
        if "db" in self.units.lower():
            self.data = utils.db2lin(self.data)
            self.units = ""

    def mask_indices(self, ind: list) -> None:
        """Masks data from given indices."""
        self.data[ind] = ma.masked

    def rebin_data(
        self, time: np.ndarray, time_new: np.ndarray, *, mask_zeros: bool = True
    ) -> list:
        """Rebins `data` in time.

        Args:
            time: 1D time array.
            time_new: 1D new time array.
            mask_zeros: Whether to mask 0 values in the returned array. Default is True.

        Returns:
            Time indices without data.

        """
        if self.data.ndim == 1:
            self.data = utils.rebin_1d(time, self.data, time_new, mask_zeros=mask_zeros)
            bad_indices = list(np.where(self.data == ma.masked)[0])
        else:
            if not isinstance(self.data, ma.MaskedArray):
                self.data = ma.masked_array(self.data)
            self.data, bad_indices = utils.rebin_2d(
                time, self.data, time_new, mask_zeros=mask_zeros
            )
        return bad_indices

    def fetch_attributes(self) -> list:
        """Returns list of user-defined attributes."""
        attributes = []
        for key, value in self.__dict__.items():
            if (
                key
                not in (
                    "variable",
                    "name",
                    "data",
                    "data_type",
                    "dimensions",
                )
                and value is not None
            ):
                attributes.append(key)
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
        if isinstance(
            self.variable,
            int | float | np.float32 | np.int8 | np.float64 | np.int32 | np.uint16,
        ):
            return np.array(self.variable)
        if isinstance(self.variable, str):
            try:
                numeric_value = utils.str_to_numeric(self.variable)
                return np.array(numeric_value)
            except ValueError:
                pass
        msg = f"Incorrect CloudnetArray input: {self.variable}"
        raise ValueError(msg)

    def _init_units(self) -> str:
        return getattr(self.variable, "units", "")

    def _init_data_type(self) -> str:
        if self.data.dtype in (np.float32, np.float64):
            return "f4"
        if self.data.dtype == np.int16:
            return "i2"
        return "i4"

    def __getitem__(self, ind: tuple) -> np.ndarray:
        return self.data[ind]

    def filter_isolated_pixels(self) -> None:
        """Filters hot pixels from radar data."""
        self._filter(utils.filter_isolated_pixels)

    def filter_vertical_stripes(self) -> None:
        """Filters vertical artifacts from radar data."""
        self._filter(utils.filter_x_pixels)

    def _filter(self, fun) -> None:
        if not isinstance(self.data, ma.MaskedArray):
            self.data = ma.masked_array(self.data)
        is_data = (~self.data.mask).astype(int)
        is_data_filtered = fun(is_data)
        self.data[is_data_filtered == 0] = ma.masked

    def calc_linear_std(self, time: np.ndarray, time_new: np.ndarray) -> None:
        """Calculates std of radar velocity.

        Args:
            time: 1D time array.
            time_new: 1D new time array.

        Notes:
            The result is masked if the bin contains masked values.
        """
        data_as_float = self.data.astype(float)
        data_as_float = ma.masked_array(data_as_float)
        self.data, _ = utils.rebin_2d(time, data_as_float, time_new, "std")

    def rebin_velocity(
        self,
        time: np.ndarray,
        time_new: np.ndarray,
        folding_velocity: np.ndarray,
    ) -> None:
        """Rebins data (Doppler velocity) in polar coordinates.

        Args:
            time: 1D time array.
            time_new: 1D new time array.
            folding_velocity: Folding velocity (m/s).
        """
        folding_scaled = ma.array(np.pi / folding_velocity)
        data_scaled = self.data * folding_scaled
        vel_x = ma.cos(data_scaled)
        vel_y = ma.sin(data_scaled)
        vel_x_mean, _ = utils.rebin_2d(time, vel_x, time_new)
        vel_y_mean, _ = utils.rebin_2d(time, vel_y, time_new)
        # Nearest neighbour interpolation for folding velocity.
        f = interp1d(
            time,
            folding_scaled,
            fill_value=np.nan,
            axis=0,
            kind="nearest",
            bounds_error=False,
        )
        folding_scaled_nearest = f(time_new)
        vel_scaled = np.arctan2(vel_y_mean, vel_x_mean)
        self.data = vel_scaled / folding_scaled_nearest
