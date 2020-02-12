"""Datasource module, containing the :class:`DataSource`
and :class:`ProfileDataSource` classes.
"""
import os
import numpy as np
import netCDF4
from cloudnetpy import RadarArray, CloudnetArray, utils


class DataSource:
    """Base class for all Cloudnet measurements and model data.

    Args:
        filename (str): Calibrated instrument / model NetCDF file.
        radar (bool, optional): Indicates if data is from cloud radar.
            Default is False.

    Attributes:
        filename (str): Filename of the input file.
        dataset (Dataset): A netCDF4 Dataset instance.
        source (str): Global attribute `source` read from the input file.
        time (MaskedArray): Time array of the instrument.
        altitude (float): Altitude of instrument above mean sea level (m).
        data (dict): Dictionary containing :class:`CloudnetArray` instances.

    """
    def __init__(self, filename, radar=False):
        self.filename = os.path.basename(filename)
        self.dataset = netCDF4.Dataset(filename)
        self.source = getattr(self.dataset, 'source', '')
        self.time = self._init_time()
        self.altitude = self._init_altitude()
        self.data = {}
        self._array_type = RadarArray if radar else CloudnetArray

    def getvar(self, *args):
        """Returns data array from the source file variables.

        Returns just the data (and no attributes) from the original variables
        dictionary, fetched from the input NetCDF file.

        Args:
            *args: possible names of the variable. The first match is returned.

        Returns:
            MaskedArray: The actual data.

        Raises:
             RuntimeError: The variable is not found.

        """
        for arg in args:
            if arg in self.dataset.variables:
                return self.dataset.variables[arg][:]
        raise RuntimeError('Missing variable in the input file.')

    def append_data(self, array, key, name=None, units=None):
        """Adds new CloudnetVariable into `data` attribute.

        Args:
            array (ndarray): Array to be added.
            key (str): Key used with *array* when added to `data` attribute
                (which is a dictionary).
            name (str, optional): CloudnetArray.name attribute. Default value
                is *key*.
            units (str, optional): CloudnetArray.units attribute.

        """
        self.data[key] = self._array_type(array, name or key, units)

    def close(self):
        """Closes the open file."""
        self.dataset.close()

    @staticmethod
    def km2m(var):
        """Converts km to m."""
        alt = var[:]
        if var.units == 'km':
            alt *= 1000
        return alt

    @staticmethod
    def m2km(var):
        """Converts m to km."""
        alt = var[:]
        if var.units == 'm':
            alt /= 1000
        return alt

    def _init_time(self):
        time = self.getvar('time')
        if max(time) > 25:
            time = utils.seconds2hours(time)
        return time

    def _init_altitude(self):
        """Returns altitude of the instrument (m)."""
        if 'altitude' in self.dataset.variables:
            altitude_above_sea = self.km2m(self.dataset.variables['altitude'])
            return float(altitude_above_sea)
        return None

    def _netcdf_to_cloudnet(self, fields):
        """Transforms netCDF4-variables into CloudnetArrays.

        Args:
            fields (tuple): netCDF4-variables to be converted. The results are
                saved in *self.data* dictionary with *fields* strings as keys.

        Notes:
            The attributes of the variables are not copied. Just the data.

        """
        for key in fields:
            self.append_data(self.dataset.variables[key], key)

    def _unknown_to_cloudnet(self, possible_names, key, units=None):
        """Transforms single netCDF4 variable into CloudnetArray.

        Args:
            possible_names(tuple): Tuple of strings containing the possible
                names of the variable in the input NetCDF file.

            key(str): Key for self.data dictionary and name-attribute for
                the saved CloudnetArray object.

            units(str, optional): Units-attribute for the CloudnetArray object.

        """
        array = self.getvar(*possible_names)
        self.append_data(array, key, units=units)


class ProfileDataSource(DataSource):
    """The :class:`ProfileDataSource` class, child of :class:`DataSource`.

    Args:
        filename (str): Raw lidar or radar file.
        radar (bool, optional): Indicates if data is from cloud radar.
            Default is False.

    Attributes:
        height (ndarray): Measurement height grid above mean sea level (m).

    """
    def __init__(self, filename, radar=False):
        super().__init__(filename, radar)
        self.height = self._get_height()

    def _get_height(self):
        """Returns height array above mean sea level (m)."""
        if 'height' in self.dataset.variables:
            return self.km2m(self.dataset.variables['height'])
        range_instrument = self.km2m(self.dataset.variables['range'])
        return np.array(range_instrument + self.altitude)
