"""Module for reading raw cloud radar data."""
import os
import netCDF4
import numpy as np
from cloudnetpy import output, utils
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.categorize import DataSource
from cloudnetpy.metadata import MetaData


def mira2nc(mmclx_file, output_file, site_meta, rebin_data=False):
    """Converts Mira cloud radar Level 1 file into NetCDF file.

    This function converts raw cloud radar file into a much smaller file that
    contains only the relevant data and can be used in further processing steps.

    Args:
        mmclx_file (str): Raw radar file in NetCDF format.
        output_file (str): Output file name.
        site_meta (dict): Dictionary containing information about the
            site. Required key value pairs are 'name'.
        rebin_data (bool, optional): If True, rebins data to 30s resolution.
            Otherwise keeps the native resolution. Default is False.

    Examples:
          >>> from cloudnetpy.instruments.mira import mira2nc
          >>> site_meta = {'name': 'Vehmasmaki'}
          >>> mira2nc('raw_radar.mmclx', 'radar.nc', site_meta)

    """
    raw_mira = Mira(mmclx_file, site_meta)
    raw_mira.linear_to_db(('Ze', 'ldr', 'SNR'))
    if rebin_data:
        snr_gain = raw_mira.rebin_fields()
    else:
        snr_gain = 1
    raw_mira.screen_by_snr(snr_gain)
    raw_mira.add_meta()
    output.update_attributes(raw_mira.data, MIRA_ATTRIBUTES)
    _save_mira(mmclx_file, raw_mira, output_file)


class Mira(DataSource):
    """Class for MIRA-36 raw radar data. Child of DataSource().

    Args:
        raw_radar_file (str): Filename of raw MIRA NetCDF file.
        site_meta (dict): Site properties in a dictionary. Required
            keys are: 'name'.

    """
    keymap = {'Zg': 'Ze',
              'VELg': 'v',
              'RMSg': 'width',
              'LDRg': 'ldr',
              'SNRg': 'SNR'}

    def __init__(self, raw_radar_file, site_meta):
        super().__init__(raw_radar_file)
        self.source = 'METEK MIRA-36'
        self.radar_frequency = 35.5
        self._init_data()
        self.range = self.getvar(self, 'range')
        self.location = site_meta['name']

    def _init_data(self):
        """Reads correct fields and fixes the names."""
        for raw_key in self.keymap:
            name = self.keymap[raw_key]
            self.data[name] = CloudnetArray(self.getvar(raw_key), name)

    @staticmethod
    def _estimate_snr_gain(time_sparse, time_dense):
        """Returns factor for SNR (dB) increase when data is binned."""
        binning_ratio = utils.mdiff(time_sparse) / utils.mdiff(time_dense)
        return np.sqrt(binning_ratio)

    def linear_to_db(self, variables_to_log):
        """Changes some linear units to logarithmic."""
        for name in variables_to_log:
            self.data[name].lin2db()

    def rebin_fields(self):
        time_grid = utils.time_grid()
        for field in self.data:
            self.data[field].rebin_data(self.time, time_grid)
        snr_gain = self._estimate_snr_gain(time_grid, self.time)
        self.time = time_grid
        return snr_gain

    def screen_by_snr(self, snr_gain=1, snr_limit=-17):
        """ Screens by SNR."""
        ind = np.where(self.data['SNR'][:] * snr_gain < snr_limit)
        for field in self.data:
            self.data[field].mask_indices(ind)

    def add_meta(self):
        self._add_geolocation()
        for key in ('time', 'range', 'radar_frequency'):
            self.data[key] = CloudnetArray(getattr(self, key), key)

    def _add_geolocation(self):
        for key in ('Latitude', 'Longitude', 'Altitude'):
            value = getattr(self.dataset, key).split()[0]
            key = key.lower()
            self.data[key] = CloudnetArray(float(value), key)


def _save_mira(mmclx_file, raw_radar, output_file):
    """Saves the MIRA radar file."""
    dims = {'time': len(raw_radar.time),
            'range': len(raw_radar.range)}
    rootgrp = output.init_file(output_file, dims, raw_radar.data)
    fields_from_raw = ('nfft', 'prf', 'nave', 'zrg', 'rg0', 'drg',
                       'NyquistVelocity')
    output.copy_variables(netCDF4.Dataset(mmclx_file), rootgrp, fields_from_raw)
    rootgrp.title = f"Radar file from {raw_radar.location}"
    rootgrp.year, rootgrp.month, rootgrp.day = _date_from_filename(raw_radar.filename)
    rootgrp.location = raw_radar.location
    rootgrp.history = f"{utils.get_time()} - radar file created"
    rootgrp.source = raw_radar.source
    rootgrp.close()


def _date_from_filename(full_path):
    """Returns date from the beginning of file name."""
    plain_file = os.path.basename(full_path)
    date = plain_file[:8]
    return date[:4], date[4:6], date[6:8]


MIRA_ATTRIBUTES = {
    'Ze': MetaData(
        long_name='Radar reflectivity factor (uncorrected), vertical polarization',
        units='dBZ',
    ),
    'SNR': MetaData(
        long_name='Signal-to-noise ratio',
        units='dB',
    )
}
