"""Module for reading / converting disdrometer data."""
import glob
from typing import Optional, Union
import logging
import netCDF4
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
from cloudnetpy.metadata import MetaData
from cloudnetpy import output
from cloudnetpy import utils
from cloudnetpy.instruments.ceilometer import Ceilometer, NoiseParam


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
        site_meta: Dictionary containing information about the site. Required keys are `name` and
            `altitude`. If the zenith angle of the instrument is NOT 5 degrees, it should be
            provided like this: {'zenith_angle': 6}.
        keep_uuid: If True, keeps the UUID of the old file, if that exists. Default is False
            when new UUID is generated.
        uuid: Set specific UUID for the file.
        date: Expected date of the measurements as YYYY-MM-DD.

    Returns:
        UUID of the generated file.

    """
    polly = PollyXt(site_meta, date)
    polly.fetch_data(input_folder)
    polly.get_date_and_time(polly.epoch)
    polly.fetch_zenith_angle()
    for key in ('depolarisation', 'beta'):
        polly.data[key] = polly.calc_screened_product(polly.data[f'{key}_raw'])
    polly.data['beta_smooth'] = polly.calc_beta_smooth(polly.data['beta'])
    polly.screen_depol()
    polly.prepare_data(site_meta)
    polly.remove_raw_data()
    polly.prepare_metadata()
    polly.data_to_cloudnet_arrays()
    attributes = output.add_time_attribute(ATTRIBUTES, polly.metadata['date'])
    output.update_attributes(polly.data, attributes)
    return _save_pollyxt(polly, output_file, keep_uuid, uuid)


class PollyXt(Ceilometer):

    noise_param = NoiseParam(n_gates=500)

    def __init__(self, site_meta: dict, expected_date: Optional[str] = None):
        super().__init__(self.noise_param)
        self.metadata = site_meta
        self.expected_date = expected_date
        self.model = 'PollyXT Raman lidar'
        self.wavelength = 1064
        self.epoch = None

    def fetch_zenith_angle(self) -> None:
        default = 5
        self.data['zenith_angle'] = self.metadata.get('zenith_angle', default)

    def fetch_data(self, input_folder: str) -> None:
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
            self.epoch = utils.get_epoch(nc_bsc['time'].unit)
            try:
                time = np.array(_read_array_from_file_pair(nc_bsc, nc_depol, 'time'))
            except AssertionError:
                _close(nc_bsc, nc_depol)
                continue
            beta_raw = nc_bsc.variables[bsc_key][:]
            depol_raw = nc_depol.variables[depol_key][:]
            for array, key in zip([beta_raw, depol_raw, time], ['beta_raw', 'depolarisation_raw',
                                                                'time']):
                self.data = utils.append_data(self.data, key, array)
            calibration_factors.append(nc_bsc.variables[bsc_key].Lidar_calibration_constant_used)
            _close(nc_bsc, nc_depol)
        self.data['calibration_factor'] = np.mean(calibration_factors)


def _read_array_from_multiple_files(files1: list, files2: list, key) -> np.ndarray:
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
                               key: str) -> np.ndarray:
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
    dims = {key: len(polly.data[key][:]) for key in ('time', 'range')}
    file_type = 'lidar'
    rootgrp = output.init_file(output_file, dims, polly.data, keep_uuid, uuid)
    uuid = rootgrp.file_uuid
    output.add_file_type(rootgrp, file_type)
    rootgrp.title = f"{file_type.capitalize()} file from {polly.metadata['name']}"
    rootgrp.year, rootgrp.month, rootgrp.day = polly.metadata['date']
    rootgrp.location = polly.metadata['name']
    rootgrp.history = f"{utils.get_time()} - {file_type} file created"
    rootgrp.source = polly.metadata['source']
    output.add_references(rootgrp)
    rootgrp.close()
    return uuid


ATTRIBUTES = {
    'depolarisation': MetaData(
        long_name='Lidar depolarisation',
        units='',
        comment='SNR screened lidar depolarisation at 532 nm.'
    ),
    'calibration_factor': MetaData(
        long_name='Backscatter calibration factor',
        comment='Mean value of the day.',
        units='',
    ),
}
