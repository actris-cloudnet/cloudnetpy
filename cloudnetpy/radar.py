"""Module for reading raw cloud radar data."""

import uuid
import numpy as np
import numpy.ma as ma
import netCDF4
from cloudnetpy import ncf
from cloudnetpy import utils
from cloudnetpy import output
import sys


class RadarVariable():
    def __init__(self, netcdf4_variable):
        self.data = netcdf4_variable[:]
        self.units = netcdf4_variable.units

    def lin2db(self):
        if 'db' not in self.units.lower():
            self.data = utils.lin2db(self.data)
            self.units = 'dB'

    def rebin_data(self, x, x_new):
        self.data = utils.rebin_2d(x, self.data, x_new)

    def mask_indices(self, ind):
        self.data[ind] = ma.masked


def mmclx2nc(mmclx_file, output_file, site_name,
             site_location, rebin_data=True):
    """Converts mmclx files to compressed NetCDF files.

    High level API to process raw cloud radar files into
    files that can be used in further processing steps.

    Args:
        mmclx_file (str): Raw radar file.
        output_file (str): Output file name.
        site_name (str): Name of the site, e.g. 'Mace-Head'.
        site_location (tuple): 3-element tuple containing site
            latitude, longitude and altitude.
        rebin_data (bool, optional): If True, rebins data to 30s resolution.
            Otherwise keeps the native resolution. Default is True.

    """
    raw_data = ncf.load_nc(mmclx_file)
    time_grid, height_grid, radar_time = _create_grid(raw_data, rebin_data)
    keymap = _change_variable_names()
    radar_data = _read_raw_data(keymap, raw_data)
    _fix_units(radar_data, ('Zh', 'ldr', 'SNR'))

    #_remove_invalid(radar_data)
    if rebin_data:
        _rebin_fields(radar_data, radar_time, time_grid)
        snr_gain = estimate_snr_gain(radar_time, time_grid)
    else:
        snr_gain = 1

    _screen_by_snr(radar_data, snr_gain)
    _add_meta(radar_data, site_location, time_grid, height_grid)
    obs = output.create_objects_for_output(radar_data)
    _save_radar(mmclx_file, output_file, obs, time_grid, height_grid, site_name)


def _screen_by_snr(radar_data, snr_gain=1, snr_limit=-17):
    """ Screens by SNR."""
    ind = np.where(radar_data['SNR'].data*snr_gain < snr_limit)
    for field in radar_data:
        radar_data[field].mask_indices(ind)

    
def _read_raw_data(keymap, raw_data):
    """Reads correct fields and fixes the names."""
    radar_data = {}
    for raw_key in keymap:
        radar_data[keymap[raw_key]] = RadarVariable(raw_data[raw_key])
    return radar_data


def _fix_units(radar_data, lin2log_list):
    """Changes some linear units to logarithmic."""
    for name in lin2log_list:
        radar_data[name].lin2db()


def _rebin_fields(radar_data, radar_time, time_grid):
    """Rebins the data."""
    for field in radar_data:
        radar_data[field].rebin_data(radar_time, time_grid)

        
def estimate_snr_gain(radar_time, time_grid):
    """Returns factor for SNR (dB) increase when data is binned."""
    binning_ratio = utils.mdiff(time_grid)/utils.mdiff(radar_time)
    return np.sqrt(binning_ratio)


def _change_variable_names():
    """ Returns mapping from radar variable names
    to names we use in Cloudnet files."""
    keymap = {'Zg':'Zh',
              'VELg':'v',
              'RMSg':'width',
              'LDRg':'ldr',
              'SNRg':'SNR'}
    return keymap


def _add_meta(radar_data, site_location, time_grid, height_grid):
    """ Add some meta data for output writing."""
    extra_vars = {
        'latitude': site_location[0],
        'longitude': site_location[1],
        'altitude': site_location[2],
        'radar_frequency': 35.5,  # not good to fix this
    }
    radar_data = {**extra_vars, **radar_data}
    radar_data['time'] = time_grid
    radar_data['range'] = height_grid


def _create_grid(raw_data, rebin_data):
    """Creates time / height grid for radar."""
    radar_time = utils.seconds2hour(raw_data['time'][:])
    if rebin_data:
        time_grid = utils.time_grid()
    else:
        time_grid = radar_time
    return time_grid, raw_data['range'][:], radar_time


def _remove_invalid(radar_data):
    """Remove invalid values from the dict."""
    for field in radar_data:
        radar_data[field] = ma.masked_invalid(radar_data[field])


def _save_radar(raw_file, output_file, obs, time, radar_range, site_name):
    """Saves the radar file."""
    raw = netCDF4.Dataset(raw_file)
    rootgrp = netCDF4.Dataset(output_file, 'w', format='NETCDF4_CLASSIC')
    rootgrp.createDimension('time', len(time))
    rootgrp.createDimension('range', len(radar_range))
    fields_from_raw = ('nfft', 'prf', 'nave', 'zrg', 'rg0', 'drg', 'lambda')
    output.copy_variables(raw, rootgrp, fields_from_raw)
    output.write_vars2nc(rootgrp, obs, zlib=True)
    # global attributes:
    #rootgrp.title = varname + ' from ' + cat.location + ', ' + get_date(cat)
    rootgrp.location = site_name
    rootgrp.institution = 'Data processed at the Finnish Meteorological Institute.'
    rootgrp.file_uuid = str(uuid.uuid4().hex)
    # copy these global attributes from categorize file
    #copy_global(cat, rootgrp, {'location', 'day', 'month', 'year', 'source', 'history'})
    rootgrp.close()
