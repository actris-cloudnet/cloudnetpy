import netCDF4
from datetime import datetime, timezone
#from cloudnetpy import ncf


class CnetVar:
    """Class for Cloudnet variables. Not sure this is needed though."""
    def __init__(self, name, data,
                 data_type='f4', size=('time','height'), zlib=True, fill_value=True,
                 long_name='', units='', comment='', plot_scale=None, plot_range=None,
                 bias_variable=None, error_variable=None, extra_attributes=None):
        self.name = name
        self.data = data
        self.data_type = data_type
        self.size = size
        self.zlib = zlib
        self.long_name = long_name
        self.units = units
        self.comment = comment
        self.plot_scale = plot_scale
        self.plot_range = plot_range
        self.extra_attributes = extra_attributes
        if (bias_variable and type(bias_variable) == bool):
            self.bias_variable = name + '_bias'
        else:
            self.bias_variable = bias_variable
        if (error_variable and type(error_variable) == bool):
            self.error_variable = name + '_error'
        else:
            self.error_variable = error_variable
        if (fill_value and type(fill_value) == bool):
            self.fill_value = netCDF4.default_fillvals[data_type]
        else:
            self.fill_value = fill_value


def write_vars2nc(rootgrp, obs):
    """Iterate over Cloudnet instances and write to given rootgrp."""
    for var in obs:
        ncvar = rootgrp.createVariable(var.name, var.data_type, var.size,
                                       zlib=var.zlib, fill_value=var.fill_value)
        ncvar[:] = var.data
        ncvar.long_name = var.long_name
        if var.units:
            ncvar.units = var.units
        if var.error_variable:
            ncvar.error_variable = var.error_variable
        if var.bias_variable:
            ncvar.bias_variable = var.bias_variable
        if var.comment:
            ncvar.comment = var.comment
        if var.plot_range:
            ncvar.plot_range = var.plot_range
        if var.plot_scale:
            ncvar.plot_scale = var.plot_scale
        if var.extra_attributes:
            for attr, value in var.extra_attributes.items():
                setattr(ncvar, attr, value)


def _copy_dimensions(file_from, file_to, dims_to_be_copied):
    """Copies dimensions from one file to another. """
    for dname, dim in file_from.dimensions.items():
        if dname in dims_to_be_copied:
            file_to.createDimension(dname, len(dim))


def _copy_variables(file_from, file_to, vars_to_be_copied):
    """Copies variables (and their attributes) from one file to another."""
    for vname, varin in file_from.variables.items():
        if vname in vars_to_be_copied:
            outVar = file_to.createVariable(vname, varin.datatype, varin.dimensions)
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})
            outVar[:] = varin[:]


def _copy_global(file_from, file_to, attrs_to_be_copied):
    """Copies global attributes from one file to another."""
    for aname in file_from.ncattrs():
        if aname in attrs_to_be_copied:
            setattr(file_to, aname, file_from.getncattr(aname))


def save_cat(file_name, time, height, model_time, model_height, obs, aux):
    rootgrp = netCDF4.Dataset(file_name, 'w', format='NETCDF4')
    # create dimensions
    time = rootgrp.createDimension('time', len(time))
    height = rootgrp.createDimension('height', len(height))
    model_time = rootgrp.createDimension('model_time', len(model_time))
    model_height = rootgrp.createDimension('model_height', len(model_height))
    # root group variables
    write_vars2nc(rootgrp, obs)
    # global attributes:
    rootgrp.Conventions = 'CF-1.7'
    rootgrp.title = 'Categorize file from ' + aux[0]
    rootgrp.institution = 'Data processed at the ' + aux[1]
    #rootgrp.year = int(dvec[:4])
    #rootgrp.month = int(dvec[5:7])
    #rootgrp.day = int(dvec[8:])
    #rootgrp.software_version = version
    #rootgrp.git_version = ncf.git_version()
    #rootgrp.file_uuid = str(uuid.uuid4().hex)
    rootgrp.references = 'https://doi.org/10.1175/BAMS-88-6-883'
    rootgrp.history = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + ' - categorize file created'
    rootgrp.close()
