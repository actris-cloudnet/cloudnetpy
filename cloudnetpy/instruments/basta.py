"""Module for reading / converting BASTA radar data."""
from typing import Union, Optional
import numpy as np
import numpy.ma as ma
import netCDF4
from cloudnetpy import output, utils, CloudnetArray
from cloudnetpy.categorize import DataSource
from cloudnetpy.metadata import MetaData


def basta2nc(basta_file: str,
             output_file: str,
             site_meta: dict,
             keep_uuid: bool = False,
             uuid: Optional[str] = None,
             date: Optional[str] = None) -> str:
    """Converts daily BASTA cloud radar file into netCDF file.

    This function converts daily BASTA file into a much smaller file that
    contains only the relevant data and can be used in further processing
    steps.

    Args:
        basta_file (str): Filename of a daily BASTA .nc file.
        output_file (str): Output filename.
        site_meta (dict): Dictionary containing information about the
            site. Required key value pair is `name`.
        keep_uuid (bool, optional): If True, keeps the UUID of the old file,
            if that exists. Default is False when new UUID is generated.
        uuid (str, optional): Set specific UUID for the file.
        date (str, optional): Expected date of the measurements as YYYY-MM-DD.

    Returns:
        str: UUID of the generated file.

    Raises:
        ValueError: Timestamps do not match the expected date.

    Examples:
          >>> from cloudnetpy.instruments import basta2nc
          >>> site_meta = {'name': 'Palaiseau'}
          >>> basta2nc('basta_file.nc', 'radar.nc', site_meta)

    """
    raw_basta = Basta(basta_file, site_meta)
    if date:
        raw_basta.validate_date(date)
    raw_basta.add_meta()
    attributes = output.add_time_attribute(ATTRIBUTES, raw_basta.date)
    output.update_attributes(raw_basta.data, attributes)
    return _save_basta(basta_file, raw_basta, output_file, keep_uuid, uuid)


class Basta(DataSource):
    """Class for BASTA raw radar data. Child of DataSource().

    Args:
        basta_file (str): BASTA netCDF filename.
        site_meta (dict): Site properties in a dictionary. Required keys are: `name`.

    """
    keymap = {'reflectivity': 'Ze',
              'velocity': 'v'}

    def __init__(self, basta_file: str, site_meta: dict):
        super().__init__(basta_file)
        self.source = 'BASTA'
        self.radar_frequency = 35.0
        self._init_data()
        self.range = self.getvar(self, 'range')
        self.location = site_meta['name']
        self.date = self.get_date()

    def add_meta(self) -> None:
        """Adds some meta-data."""
        for key in ('time', 'range'):
            self.data[key] = CloudnetArray(getattr(self, key), key)
        self._unknown_to_cloudnet(('ambiguous_velocity',), 'nyquist_velocity')
        self._unknown_to_cloudnet(('carrier_frequency',), 'radar_frequency')

    def validate_date(self, expected_date: str) -> None:
        """Validates expected data."""
        date_units = self.dataset.variables['time'].units
        date = date_units.split()[2]
        if expected_date != date:
            raise ValueError('Basta date not what expected.')

    def _init_data(self) -> None:
        """Reads correct fields and fixes the names."""
        for raw_key in self.keymap:
            name = self.keymap[raw_key]
            array = self.getvar(raw_key)
            array[~np.isfinite(array)] = ma.masked
            self.data[name] = CloudnetArray(array, name)


def _save_basta(basta_file: str,
                raw_radar: Basta,
                output_file: str,
                keep_uuid: Union[bool, None],
                uuid: Union[str, None]) -> str:

    dims = {'time': len(raw_radar.time),
            'range': len(raw_radar.range)}

    rootgrp = output.init_file(output_file, dims, raw_radar.data, keep_uuid, uuid)
    uuid = rootgrp.file_uuid
    fields_from_raw = ['elevation', 'pulse_width']
    output.copy_variables(netCDF4.Dataset(basta_file), rootgrp, fields_from_raw)
    output.add_file_type(rootgrp, 'radar')
    rootgrp.title = f"Radar file from {raw_radar.location}"
    rootgrp.year, rootgrp.month, rootgrp.day = raw_radar.date
    rootgrp.location = raw_radar.location
    rootgrp.history = f"{utils.get_time()} - radar file created"
    rootgrp.source = raw_radar.source
    output.add_references(rootgrp)
    rootgrp.close()
    return uuid


ATTRIBUTES = {
    'Ze': MetaData(
        long_name='Radar reflectivity factor',
        units='dBZ',
    ),
}
