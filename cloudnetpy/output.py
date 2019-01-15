""" Functions for Categorize output file writing."""
from datetime import datetime, timezone
import uuid
import netCDF4
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils
from cloudnetpy import config
from cloudnetpy.metadata import ATTRIBUTES


class CloudnetVariable():
    """Creates Cloudnet variables from NetCDF variables.

    Attributes:
        _name (str): Name of the variable.
        _data (array_like): The actual data.
        _data_type (str): 'i4' for integers, 'f4' for floats.
        _units (str): Copied from the original netcdf4
            variable (if existing).

    Notes:
        Attributes starting with an underscore serve special purpose
        in output file writing and processing. Thus, they are separated
        from optional normal attributes such as self.comment, self.history
        and so on, that can be attached to the variable in various
        processing steps.

    """

    def __init__(self, netcdf4_variable, name):
        self._name = name
        self._data = netcdf4_variable[:]
        self._data_type = self._init_data_type()
        self._init_units(netcdf4_variable)
        self.units = None

    def _init_units(self, netcdf4_variable):
        if hasattr(netcdf4_variable, 'units'):
            self.units = netcdf4_variable.units

    def _init_data_type(self):
        if isinstance(self._data, int):
            return 'i4'
        return 'f4'

    def lin2db(self):
        """Converts linear units do log."""
        if 'db' not in self.units.lower():
            self._data = utils.lin2db(self._data)
            self.units = 'dB'

    def db2lin(self):
        """Converts log units to linear."""
        if 'db' in self.units.lower():
            self._data = utils.db2lin(self._data)
            self.units = ''

    def rebin_data(self, time, time_new, height=None, height_new=None):
        """Rebins data in time and optionally in height."""
        self._data = utils.rebin_2d(time, self._data, time_new)
        if np.any(height) and np.any(height_new):
            self._data = utils.rebin_2d(height, self._data.T, height_new).T

    def rebin_in_polar(self, time, time_new, folding_velocity):
        """Rebins velocity in polar coordinates."""
        scaled = self._data * folding_velocity
        velo_x, velo_y = np.cos(scaled), np.sin(scaled)
        velo_x_mean = utils.rebin_2d(time, velo_x, time_new)
        velo_y_mean = utils.rebin_2d(time, velo_y, time_new)
        self._data = np.arctan2(velo_y_mean, velo_x_mean) / folding_velocity

    def mask_indices(self, ind):
        """Masks data from given indices."""
        self._data[ind] = ma.masked

    def fetch_attributes(self):
        """Returns list of user-defined attributes."""
        return (x for x in self.__dict__.keys() if not x.startswith('_'))

    def set_attributes(self, attributes):
        for key in attributes._fields:
            data = getattr(attributes, key)
            if data:
                setattr(self, key, data)


def write_vars2nc(rootgrp, cnet_variables, zlib):
    """Iterate over Cloudnet instances and write to given rootgrp."""

    def _get_dimensions(array):
        """Finds correct dimensions for a variable."""
        if not hasattr(array, '__len__') or len(array) == 1:
            return ()
        size = ()
        file_dims = rootgrp.dimensions
        array_dims = array.shape
        for length in array_dims:
            dim = [key for key in file_dims.keys() if file_dims[key].size == length][0]
            size = size + (dim,)
        return size

    for name in cnet_variables:
        obj = cnet_variables[name]
        size = _get_dimensions(obj._data)
        ncvar = rootgrp.createVariable(obj._name, obj._data_type, size, zlib=zlib)
        ncvar[:] = obj._data
        for attr in obj.fetch_attributes():
            setattr(ncvar, attr, getattr(obj, attr))


def save_cat(file_name, time, height, model_time, model_height, obs, radar_meta, zlib):
    """Creates a categorize netCDF4 file and saves all data into it."""
    rootgrp = netCDF4.Dataset(file_name, 'w', format='NETCDF4_CLASSIC')
    # create dimensions
    time = rootgrp.createDimension('time', len(time))
    height = rootgrp.createDimension('height', len(height))
    model_time = rootgrp.createDimension('model_time', len(model_time))
    model_height = rootgrp.createDimension('model_height', len(model_height))
    # root group variables
    write_vars2nc(rootgrp, obs, zlib)
    # global attributes:
    rootgrp.Conventions = 'CF-1.7'
    rootgrp.title = 'Categorize file from ' + radar_meta['location']
    rootgrp.institution = 'Data processed at the ' + config.INSTITUTE
    dvec = radar_meta['date']
    rootgrp.year = int(dvec[:4])
    rootgrp.month = int(dvec[5:7])
    rootgrp.day = int(dvec[8:])
    #rootgrp.software_version = version
    #rootgrp.git_version = ncf.git_version()
    rootgrp.file_uuid = str(uuid.uuid4().hex)
    rootgrp.references = 'https://doi.org/10.1175/BAMS-88-6-883'
    rootgrp.history = f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} - categorize file created"
    rootgrp.close()


def status_name(long_name):
    """ Default retrieval status variable name """
    return long_name + ' retrieval status'


def bias_name(long_name):
    """ Default bias variable name """
    return 'Possible bias in ' + long_name.lower() + ', one standard deviation'


def err_name(long_name):
    """ Default error variable name """
    return 'Random error in ' + long_name.lower() + ', one standard deviation'


def err_comm(long_name):
    """ Default error comment """
    return ('This variable is an estimate of the one-standard-deviation random error\n'
            'in ' + long_name.lower() + 'due to the uncertainty of the retrieval, including\n'
            'the random error in the radar and lidar parameters.')


def bias_comm(long_name):
    """ Default bias comment """
    return ('This variable is an estimate of the possible systematic error in '
            + long_name.lower() + 'due to the\n'
            'uncertainty in the calibration of the radar and lidar.')


def anc_names(var, bias=False, err=False, sens=False):
    """Returns list of ancillary variable names."""
    out = ''
    if bias:
        out += f"{var}_bias "
    if err:
        out += f"{var}_error "
    if sens:
        out += f"{var}_sensitivity "
    return out[:-1]


def copy_dimensions(file_from, file_to, dims_to_be_copied):
    """Copies dimensions from one file to another. """
    for dname, dim in file_from.dimensions.items():
        if dname in dims_to_be_copied:
            file_to.createDimension(dname, len(dim))


def copy_variables(file_from, file_to, vars_to_be_copied):
    """Copies variables (and their attributes) from one file to another."""
    for vname, varin in file_from.variables.items():
        if vname in vars_to_be_copied:
            varout = file_to.createVariable(vname, varin.datatype, varin.dimensions)
            varout.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            varout[:] = varin[:]


def copy_global(file_from, file_to, attrs_to_be_copied):
    """Copies global attributes from one file to another."""
    for aname in file_from.ncattrs():
        if aname in attrs_to_be_copied:
            setattr(file_to, aname, file_from.getncattr(aname))
