import numpy as np
import numpy.ma as ma
from cloudnetpy import ncf
from cloudnetpy import utils
from cloudnetpy import plotting
from cloudnetpy import output
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import netCDF4


def mmclx2nc(mmclx_file, output_file, site_name,
             site_location, rebin_data=True):
    """Converts mmclx files to compressed NetCDF files.

    Reads raw cloud radar NetCDF files, converts the units,
    and saves the essential fields to new file.

    Args:
        mmclx_file (str): Raw radar file.
        output_file (str): Output file name.
        site_name (str): Name of the site, e.g. 'Mace-Head'.
        site_location (tuple): 3-element tuple containing site 
            latitude, longitude and altitude.
        rebin_data (bool, optional): If True, rebins data to 30s resolution. 
            Otherwise keeps the native resolution. Default is True. 

    """    
    radar_data = ncf.load_nc(mmclx_file)
    radar_time = utils.seconds2hour(radar_data['time'][:])    
    if rebin_data:
        time_new = utils.time_grid()
    else:
        time_new = radar_time
    radar_range = radar_data['range'][:]
    
    # We want different field names
    fields = {'Zg':'Z',
              'VELg':'v',
              'RMSg':'width',
              'LDRg':'ldr',
              'SNRg':'SNR'}
    data_fields = {}
    for field in fields:
        data = radar_data[field][:]
        data = ma.masked_invalid(data)
        if rebin_data:
            data = utils.rebin_2d(radar_time, data, time_new)
        data_fields[fields[field]] = data

    # SNR-screening
    SNR_LIMIT = -17
    ind = np.where(utils.lin2db(data_fields['SNR']) < SNR_LIMIT)
    for field in data_fields:
        data = data_fields[field]
        data.mask[ind] = True
        data_fields[field] = data

    extra_vars = {
        'latitude': site_location[0],
        'longitude': site_location[1],
        'altitude': site_location[2],
        'radar_frequency': 35.5,
    }
    output_data = {**extra_vars, **data_fields}
    output_data['time'] = time_new
    output_data['range'] = radar_range

    obs = output.create_objects_for_output(output_data)
    save_radar(mmclx_file, output_file, obs, time_new, radar_range)

    
def save_radar(raw_file, output_file, obs, time, radar_range):

    raw = netCDF4.Dataset(raw_file)
    rootgrp = netCDF4.Dataset(output_file, 'w', format='NETCDF4_CLASSIC')

    rootgrp.createDimension('time', len(time))
    rootgrp.createDimension('range', len(radar_range))

    vars_copied_from_raw = ('nfft', 'prf', 'nave', 'zrg',
                            'rg0', 'drg', 'lambda')
    
    output.copy_variables(raw, rootgrp, vars_copied_from_raw)
    
    # write variables into file
    output.write_vars2nc(rootgrp, obs, zlib=True)
    # global attributes:
    #rootgrp.title = varname + ' from ' + cat.location + ', ' + get_date(cat)
    #rootgrp.institution = 'Data processed at the Finnish Meteorological Institute.'
    #rootgrp.file_uuid = str(uuid.uuid4().hex)
    # copy these global attributes from categorize file
    #copy_global(cat, rootgrp, {'location', 'day', 'month', 'year', 'source', 'history'})
    rootgrp.close()
