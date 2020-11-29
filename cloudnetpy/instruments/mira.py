"""Module for reading raw cloud radar data."""
from typing import Union
import netCDF4
import numpy as np
import numpy.ma as ma
from cloudnetpy import output, utils, CloudnetArray
from cloudnetpy.categorize import DataSource
from cloudnetpy.metadata import MetaData


def mira2nc(mmclx_file: str,
            output_file: str,
            site_meta: dict,
            rebin_data: bool = False,
            keep_uuid: bool = False,
            uuid: Union[str, None] = None) -> str:
    """Converts METEK MIRA-35 cloud radar Level 1 file into netCDF file.

    This function converts raw cloud radar file into a much smaller file that
    contains only the relevant data and can be used in further processing
    steps.

    Args:
        mmclx_file (str): Raw radar file in netCDF format.
        output_file (str): Output file name.
        site_meta (dict): Dictionary containing information about the
            site. Required key value pair is `name`.
        rebin_data (bool, optional): If True, rebins data to 30s resolution.
            Otherwise keeps the native resolution. Default is False.
        keep_uuid (bool, optional): If True, keeps the UUID of the old file,
            if that exists. Default is False when new UUID is generated.
        uuid (str, optional): Set specific UUID for the file.
    
    Returns:
        str: UUID of the generated file.

    Examples:
          >>> from cloudnetpy.instruments import mira2nc
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
    return _save_mira(mmclx_file, raw_mira, output_file, keep_uuid, uuid)


class Mira(DataSource):
    """Class for MIRA-35 raw radar data. Child of DataSource().

    Args:
        raw_radar_file (str): Filename of raw MIRA NetCDF file.
        site_meta (dict): Site properties in a dictionary. Required
            keys are: `name`.

    """
    keymap = {'Zg': 'Ze',
              'VELg': 'v',
              'RMSg': 'width',
              'LDRg': 'ldr',
              'SNRg': 'SNR'}

    def __init__(self, raw_radar_file, site_meta):
        super().__init__(raw_radar_file)
        self.source = 'METEK MIRA-35'
        self.radar_frequency = 35.5
        self._init_data()
        self.range = self.getvar(self, 'range')
        self.location = site_meta['name']

    def _init_data(self):
        """Reads correct fields and fixes the names."""
        for raw_key in self.keymap:
            name = self.keymap[raw_key]
            array = self.getvar(raw_key)
            array[~np.isfinite(array)] = ma.masked
            self.data[name] = CloudnetArray(array, name)

    def linear_to_db(self, variables_to_log):
        """Changes linear units to logarithmic."""
        for name in variables_to_log:
            self.data[name].lin2db()

    def screen_by_snr(self, snr_gain=1, snr_limit=-17):
        """Screens by SNR."""
        ind = np.where(self.data['SNR'][:] * snr_gain < snr_limit)
        for field in self.data:
            self.data[field].mask_indices(ind)

    def add_meta(self):
        """Adds some meta-data."""
        self._add_geolocation()
        for key in ('time', 'range', 'radar_frequency'):
            self.data[key] = CloudnetArray(getattr(self, key), key)
        self._unknown_to_cloudnet(('NyquistVelocity',), 'nyquist_velocity')

    def _add_geolocation(self):
        for key in ('Latitude', 'Longitude', 'Altitude'):
            value = getattr(self.dataset, key).split()[0]
            key = key.lower()
            self.data[key] = CloudnetArray(float(value), key)

    def rebin_fields(self):
        """Rebins fields."""
        time_grid = utils.time_grid()
        for field in self.data:
            self.data[field].rebin_data(self.time, time_grid)
        snr_gain = self._estimate_snr_gain(time_grid, self.time)
        self.time = time_grid
        return snr_gain

    @staticmethod
    def _estimate_snr_gain(time_sparse, time_dense):
        """Returns factor for SNR (dB) increase when data is binned."""
        binning_ratio = utils.mdiff(time_sparse) / utils.mdiff(time_dense)
        return np.sqrt(binning_ratio)


def _save_mira(mmclx_file, raw_radar, output_file, keep_uuid, uuid: Union[str, None]):
    """Saves the MIRA radar file."""
    dims = {'time': len(raw_radar.time),
            'range': len(raw_radar.range)}
    rootgrp = output.init_file(output_file, dims, raw_radar.data, keep_uuid, uuid)
    uuid = rootgrp.file_uuid
    fields_from_raw = ('nfft', 'prf', 'nave', 'zrg', 'rg0', 'drg')
    output.copy_variables(netCDF4.Dataset(mmclx_file), rootgrp, fields_from_raw)
    output.add_file_type(rootgrp, 'radar')
    rootgrp.title = f"Radar file from {raw_radar.location}"
    rootgrp.year, rootgrp.month, rootgrp.day = _find_measurement_date(raw_radar)
    rootgrp.location = raw_radar.location
    rootgrp.history = f"{utils.get_time()} - radar file created"
    rootgrp.source = raw_radar.source
    output.add_references(rootgrp)
    rootgrp.close()
    return uuid


def _find_measurement_date(raw_radar):
    """Finds measurement date from the netCDF global attribute 'source'"""
    source = raw_radar.dataset.source
    date = source.split('_')[0]
    return f"20{date[:2]}", date[2:4], date[4:6]


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
