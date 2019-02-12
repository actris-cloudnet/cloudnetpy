from subprocess import check_output 
from numpy import tile
import numpy as np
import netCDF4
import uuid
"""
Muutetaan tämä vastaamaan enemmän cloudnetpy:n output moduulia.
Tällöin saadaan se todennäköisesti pätemään varsin hyvin muillekin
datoille, joita käsitellään ja talletetaan uuteen muotoon.

Oletettavaa on, että tässäkin tiedostossa on paljon turhaa, joka on
nykyään toteutettu jollain toisella tavalla, pitää selvittää.
"""

#löytyy outputista
def copy_dimensions(file_from, file_to, dims_to_be_copied):
    """ copy nc dimensions from one file to another """
    for dname, the_dim in file_from.dimensions.items():
        if dname in dims_to_be_copied:
            file_to.createDimension(dname, len(the_dim))


# löytyy outputista
def copy_variables(file_from, file_to, vars_to_be_copied):
    """ copy nc variables (and their attributes) from one file to another """
    for vname, varin in file_from.variables.items():
        if vname in vars_to_be_copied:
            outVar = file_to.createVariable(vname, varin.datatype, varin.dimensions)            
            outVar.setncatts({k: varin.getncattr(k) for k in varin.ncattrs()})      
            outVar[:] = varin[:]


# löytyy outputista
def copy_global(file_from, file_to, attrs_to_be_copied):
    """ copy nc global attributes from one file to another """
    for aname in file_from.ncattrs():
        if aname in attrs_to_be_copied:
            setattr(file_to,aname,file_from.getncattr(aname))

            
def get_dimensions(cat):
    """ get dimensions_of categorize file """
    nalt = len(cat.dimensions['height'])
    ntime = len(cat.dimensions['time'])
    nmodel = len(cat.dimensions['model_height'])
    return (ntime, nalt, nmodel)


def get_date(cat):
    """ Return measurement date in format yyyy-mm-dd """
    return '-'.join([str(cat.year), str(cat.month).zfill(2), str(cat.day).zfill(2)])

    
def get_radar_freq(vrs, field='radar_frequency'):
    """ Read and return radar frequency """
    freq = vrs[field][:]
    if 30 < freq < 40:
        is35 = True
    elif 90 < freq < 100:
        is35 = False
    else:
        is35 = None
    return freq, is35


def git_version():
    """ Return human readable version number of the GIT repository """
    return check_output(["git", "describe", "--always"]).strip()


def expand_to_alt(x, nalt):
    """ Repeat array to all altitudes """
    return tile(x, (nalt,1)).T


class CnetVar:
    """ Class for Cloudnet retrieved variables """
    def __init__(self, name, data, sizes,
                 data_type='f4', zlib=True, fill_value=True,
                 long_name='', units='', comment='', bias_variable=None, error_variable=None, extra_attributes=None):

        size = ('time', 'height')
        if sizes == '1d':
            size = 'time'

        self.name = name
        self.data = data
        self.size = size
        self.data_type = data_type
        self.zlib = zlib
        self.long_name = long_name
        self.units = units
        self.comment = comment
        self.extra_attributes = extra_attributes
        # bias variable:
        if (bias_variable and type(bias_variable) == bool): # True
            self.bias_variable = name + '_bias'
        else:
            self.bias_variable = bias_variable
        # error variable:
        if (error_variable and type(error_variable) == bool): # True
            self.error_variable = name + '_error'
        else:
            self.error_variable = error_variable
        # fill value:
        if (fill_value and type(fill_value) == bool): # True
            self.fill_value = netCDF4.default_fillvals[data_type]
        else:
            self.fill_value = fill_value

# jossain muodossaan löytyy outputista
def write_vars2nc(rootgrp, obs):
    """ Iterate over Cloudnet instances and write to given rootgrp """
    for key, var in obs.items():
        ncvar = rootgrp.createVariable(var.name, var.data_type, var.size, zlib=var.zlib, fill_value=var.fill_value)
        ncvar[:] = var.data
        ncvar.long_name = var.long_name
        if var.units : ncvar.units = var.units
        if var.error_variable : ncvar.error_variable = var.error_variable
        if var.bias_variable : ncvar.bias_variable = var.bias_variable
        if var.comment : ncvar.comment = var.comment
        # iterate Dict of (possible) extra attributes
        if var.extra_attributes:
            for attr, value in var.extra_attributes.items():
                setattr(ncvar, attr, value)
                

# Myös löytyy outputista jossain muodossa
def save_Cnet(data, fname, varname, version):
    """ open netcdf file and write data into it 
    Works for all Cloudnet variables """

    dims = {'time': len(data.time),
            'height': len(data.height)}

    rootgrp = netCDF4.Dataset(fname, 'w', format='NETCDF4')
    #copy_dimensions(data.dataset, rootgrp, {'time', 'height'})
    copy_variables(data.dataset, rootgrp, {'altitude', 'latitude', 'longitude', 'time', 'height'})
    # write variables into file
    write_vars2nc(rootgrp, data.data)
    # global attributes:
    rootgrp.title = varname + ' from ' + data.dataset.location + ', ' + get_date(data.dataset)
    rootgrp.institution = 'Data processed at the Finnish Meteorological Institute.'
    rootgrp.software_version = version
    rootgrp.git_version = git_version()
    rootgrp.file_uuid = str(uuid.uuid4().hex)
    # copy these global attributes from categorize file
    copy_global(data.dataset, rootgrp, {'Conventions', 'location', 'day', 'month', 'year', 'source', 'history'})
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
    return ('This variable is an estimate of the one-standard-deviation random error in ' + long_name.lower() + '\n',
            'due to the uncertainty of the retrieval, including the random error in the radar and lidar parameters.')
            

def bias_comm(long_name):
    """ Default bias comment """
    return ('This variable is an estimate of the possible systematic error (one-standard-deviation) in ' + long_name.lower() + '\n',
            'due to the uncertainty in the calibration of the radar and lidar.')


def npoints(x, dist):
    """ Gives number points to cover dist in equally spaced vector x
    (x and dist must have same units)
    """
    dx = med_diff(x)
    return int(np.ceil((dist/dx)))


def med_diff(x):
    """ Gitves the median difference in vector x """
    return(np.median(np.diff(x)))


def tops_and_bases(y):
    """ From binary vector y = [0,0,1,1,0,0...] find all "islands" of ones, i.e. their starting and ending indices """
    zero = np.zeros(1)
    y2 = np.concatenate((zero, y, zero))
    y2_diff = np.diff(y2)
    bases = np.where(y2_diff == 1)[0]
    tops = np.where(y2_diff == -1)[0] - 1
    return (bases, tops)
