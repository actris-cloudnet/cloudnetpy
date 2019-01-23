"""Module for reading raw cloud radar data."""

import numpy as np
from cloudnetpy import utils
from cloudnetpy import output
from cloudnetpy.cloudnetarray import CloudnetArray
import netCDF4


def mmclx2nc(mmclx_file, output_file, site_name, site_location,
             rebin_data=False):
    """High-level API to convert mmclx file into compressed NetCDF file.

    This function converts raw cloud radar file into a much smaller file that
    contains only the relevant data and can be used in further processing steps.

    Args:
        mmclx_file (str): Raw radar file in NetCDF format.
        output_file (str): Output file name.
        site_name (str): Name of the site, e.g. 'Mace-Head'.
        site_location (tuple): 3-element tuple containing site
            latitude, longitude and altitude.
        rebin_data (bool, optional): If True, rebins data to 30s resolution.
            Otherwise keeps the native resolution. Default is False.

    Examples:
          >>> from cloudnetpy.radar import mmclx2nc
          >>> mmclx2nc('raw_radar.mmclx', 'output.nc', 'Vehmasmäki', (62.74, 27.54, 155))

    """
    raw_data = netCDF4.Dataset(mmclx_file).variables
    time_grid, height_grid, radar_time = _create_grid(raw_data, rebin_data)
    keymap = _map_variable_names()
    radar_data = _read_raw_data(keymap, raw_data)
    _linear_to_db(radar_data, ('Zh', 'ldr', 'SNR'))
    if rebin_data:
        _rebin_fields(radar_data, radar_time, time_grid)
        snr_gain = _estimate_snr_gain(radar_time, time_grid)
    else:
        snr_gain = 1
    _screen_by_snr(radar_data, snr_gain)
    _add_meta(radar_data, site_location, time_grid, height_grid)
    output.update_attributes(radar_data)
    _save_radar(mmclx_file, output_file, radar_data, time_grid,
                height_grid, site_name)


def _screen_by_snr(radar_data, snr_gain=1, snr_limit=-17):
    """ Screens by SNR."""
    ind = np.where(radar_data['SNR'].data*snr_gain < snr_limit)
    for field in radar_data:
        radar_data[field].mask_indices(ind)

    
def _read_raw_data(keymap, raw_data):
    """Reads correct fields and fixes the names.

    Args:
        keymap (dict): Fieldnames to be read from the NetCDF file
            and the corresponding names we use instead in the Cloudnet 
            processing. E.g. {'Zg': 'Zh', 'LDRg', 'ldr'}.
        raw_data (dict): NetCDF variables to be read.

    Returns:
        dict: Raw data as CloudnetVariable instances.

    """
    radar_data = {}
    for raw_key in keymap:
        name = keymap[raw_key]
        radar_data[name] = CloudnetArray(raw_data[raw_key], name)
    return radar_data


def _linear_to_db(radar_data, variables_to_log):
    """Changes some linear units to logarithmic."""    
    for name in variables_to_log:
        radar_data[name].lin2db()
    return radar_data


def _rebin_fields(radar_data, radar_time, time_grid):
    """Rebins the data."""
    for field in radar_data:
        radar_data[field].rebin_data(radar_time, time_grid)


def _estimate_snr_gain(radar_time, time_grid):
    """Returns factor for SNR (dB) increase when data is binned."""
    binning_ratio = utils.mdiff(time_grid)/utils.mdiff(radar_time)
    return np.sqrt(binning_ratio)


def _map_variable_names():
    """ Returns mapping from radar variable names
    to names we use in Cloudnet files.

    Notes:
        These names probably depend on the radar / firmware.
    """
    keymap = {'Zg': 'Zh',
              'VELg': 'v',
              'RMSg': 'width',
              'LDRg': 'ldr',
              'SNRg': 'SNR'}
    return keymap


def _add_meta(radar_data, site_location, time_grid, height_grid):
    """ Add some meta data for output writing."""
    def _add2dict(data, name):
        radar_data[name] = CloudnetArray(data, name)

    for i, key in enumerate(('latitude', 'longitude', 'altitude')):
        _add2dict(np.array(site_location[i]), key)
    _add2dict(time_grid, 'time')
    _add2dict(height_grid, 'range')
    _add2dict(35.5, 'radar_frequency')


def _create_grid(raw_data, rebin_data):
    """Creates time / height grid for radar."""
    radar_time = utils.seconds2hour(raw_data['time'][:])
    if rebin_data:
        time_grid = utils.time_grid()
    else:
        time_grid = radar_time
    return time_grid, raw_data['range'][:], radar_time


def _date_from_filename(full_path):
    """Returns date from the beginning of file name."""
    plain_file = os.path.basename(full_path)
    date = plain_file[:8]
    return date[:4], date[4:6], date[6:8]


def _save_radar(raw_file, output_file, obs, time, radar_range, site_name):
    """Saves the radar file."""
    dims = {'time': len(time), 'range': len(radar_range)}
    rootgrp = output.init_file(output_file, dims, obs, zlib=True)
    fields_from_raw = ('nfft', 'prf', 'nave', 'zrg', 'rg0', 'drg',
                       'NyquistVelocity')
    output.copy_variables(netCDF4.Dataset(raw_file), rootgrp, fields_from_raw)
    rootgrp.title = f"Radar file from {site_name}"
    rootgrp.year, rootgrp.month, rootgrp.day = _date_from_filename(raw_file)
    rootgrp.location = site_name
    rootgrp.history = f"{utils.get_time()} - radar file created"
    rootgrp.source = 'MIRA-36 cloud radar'
    rootgrp.close()
