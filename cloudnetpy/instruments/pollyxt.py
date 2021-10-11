"""Module for reading / converting disdrometer data."""
import glob
from typing import Optional, Union
import logging
import netCDF4
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.metadata import MetaData
from cloudnetpy import output
from cloudnetpy import utils


def pollyxt2nc(input_folder: str,
               output_file: str,
               site_meta: dict,
               keep_uuid: Optional[bool] = False,
               uuid: Optional[str] = None,
               date: Optional[str] = None) -> str:
    """"

    Args:
        input_folder: Filename of pollyxt file.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key is `name`.
            If the tilt angle of the instrument is NOT 5 degrees, it should be provided like this:
            {'tilt_angle': 6}.
        keep_uuid: If True, keeps the UUID of the old file, if that exists. Default is False
            when new UUID is generated.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
        UUID of the generated file.

    """
    polly = PollyXt(site_meta, date)
    polly.fetch_data(input_folder)
    polly.handle_time()
    polly.prepare_data()
    attributes = output.add_time_attribute(ATTRIBUTES, polly.date)
    output.update_attributes(polly.data, attributes)
    return _save_pollyxt(polly, output_file, keep_uuid, uuid)


class PollyXt:

    wavelength = 1064

    def __init__(self, site_metadata: dict, expected_date: Union[str, None]):
        self.site_metadata = site_metadata
        self.expected_date = expected_date
        self.source = 'PollyXT Raman Lidar'
        self.data = {}
        self.tilt_angle = site_metadata.get('tilt_angle', 5)
        self._epoch = None
        self.date = None

    def fetch_data(self, input_folder: str):
        """Read input data."""
        bsc_files = [file for file in glob.glob(f'{input_folder}/*[0-9]_att*.nc')]
        depol_files = [file for file in glob.glob(f'{input_folder}/*[0-9]_vol*.nc')]
        bsc_files.sort()
        depol_files.sort()
        if not bsc_files:
            logging.info('No pollyxt files found')
            return
        if len(bsc_files) != len(depol_files):
            logging.info('Inconsistent number of pollyxt bsc / depol files')
            return
        self.data['range'] = _read_array_from_multiple_files(bsc_files, depol_files, 'height')
        calibration_factors = []
        bsc_key = 'attenuated_backscatter_1064nm'
        depol_key = 'volume_depolarization_ratio_532nm'
        for ind, (bsc_file, depol_file) in enumerate(zip(bsc_files, depol_files)):
            nc_bsc = netCDF4.Dataset(bsc_file, 'r')
            nc_depol = netCDF4.Dataset(depol_file, 'r')
            self._epoch = utils.get_epoch(nc_bsc['time'].unit)
            try:
                time = np.array(_read_array_from_file_pair(nc_bsc, nc_depol, 'time'))
            except AssertionError:
                _close(nc_bsc, nc_depol)
                continue
            quality_mask = nc_bsc.variables['quality_mask_1064nm'][:]
            beta = ma.masked_where(quality_mask != 0, nc_bsc.variables[bsc_key][:])
            vol_depol = ma.masked_where(quality_mask != 0, nc_depol.variables[depol_key][:])
            for array, key in zip([beta, vol_depol, time], ['beta', 'vol_depol', 'time']):
                self._append_data(array, key)
            calibration_factors.append(nc_bsc.variables[bsc_key].Lidar_calibration_constant_used)
            _close(nc_bsc, nc_depol)
        self.data['calibration_factor'] = np.mean(calibration_factors)

    def prepare_data(self):
        """Add some additional data / metadata and convert into CloudnetArrays."""
        self.data['height'] = self.data['range'] * np.cos(np.radians(self.tilt_angle))
        self.data['wavelength'] = self.wavelength
        for key in self.data.keys():
            self.data[key] = CloudnetArray(self.data[key], name=key)

    def _append_data(self, array: np.array, key: str) -> None:
        if key not in self.data:
            self.data[key] = array
        else:
            self.data[key] = ma.concatenate((self.data[key], array))

    def handle_time(self):
        if self.expected_date is not None:
            self.data = utils.screen_by_time(self.data, self._epoch, self.expected_date)
        self.date = utils.seconds2date(self.data['time'][0], epoch=self._epoch)[:3]
        self.data['time'] = utils.seconds2hours(self.data['time'])


def _read_array_from_multiple_files(files1: list, files2: list, key) -> np.array:
    array = np.array([])
    for ind, (file1, file2) in enumerate(zip(files1, files2)):
        nc1 = netCDF4.Dataset(file1, 'r')
        nc2 = netCDF4.Dataset(file2, 'r')
        array1 = _read_array_from_file_pair(nc1, nc2, key)
        if ind == 0:
            array = array1
        _close(nc1, nc2)
        assert_array_equal(array, array1)
    return np.array(array)


def _read_array_from_file_pair(nc_file1: netCDF4.Dataset,
                               nc_file2: netCDF4.Dataset,
                               key: str) -> np.array:
    array1 = nc_file1.variables[key][:]
    array2 = nc_file2.variables[key][:]
    assert_array_equal(array1, array2)
    return array1


def _close(*args) -> None:
    for arg in args:
        arg.close()


def _save_pollyxt(polly: PollyXt,
                  output_file: str,
                  keep_uuid: bool,
                  uuid: Union[str, None]) -> str:
    """Saves the RPG radar / mwr file."""

    dims = {key: len(polly.data[key][:]) for key in ('time', 'range')}
    file_type = 'lidar'
    rootgrp = output.init_file(output_file, dims, polly.data, keep_uuid, uuid)
    file_uuid = rootgrp.file_uuid
    output.add_file_type(rootgrp, file_type)
    rootgrp.title = f"{file_type.capitalize()} file from {polly.site_metadata['name']}"
    rootgrp.year, rootgrp.month, rootgrp.day = polly.date
    rootgrp.location = polly.site_metadata['name']
    rootgrp.history = f"{utils.get_time()} - {file_type} file created"
    rootgrp.source = polly.source
    output.add_references(rootgrp)
    rootgrp.close()
    return file_uuid


ATTRIBUTES = {
    'vol_depol': MetaData(
        long_name='Volume depolarisation ratio at 532 nm',
        units='',
    ),
    'calibration_factor': MetaData(
        long_name='Backscatter calibration factor',
        comment='Mean value of the day',
        units='',
    ),

}
