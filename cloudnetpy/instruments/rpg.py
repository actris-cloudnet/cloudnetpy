"""This module contains RPG Cloud Radar related functions."""
from typing import Union, Tuple, Optional, List
import logging
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils, output, CloudnetArray, RadarArray
from cloudnetpy.metadata import MetaData
from cloudnetpy.instruments.rpg_reader import Fmcw94Bin, HatproBin
from cloudnetpy.exceptions import InconsistentDataError, ValidTimeStampError


def rpg2nc(path_to_l1_files: str,
           output_file: str,
           site_meta: dict,
           keep_uuid: Optional[bool] = False,
           uuid: Optional[str] = None,
           date: Optional[str] = None) -> Tuple[str, list]:
    """Converts RPG FMCW-94 cloud radar data into Cloudnet Level 1b netCDF file.

    This function reads one day of RPG Level 1 cloud radar binary files,
    concatenates the data and writes a netCDF file.

    Args:
        path_to_l1_files: Folder containing one day of RPG LV1 files.
        output_file: Output file name.
        site_meta: Dictionary containing information about the
            site. Required key value pairs are `altitude` (metres above mean
            sea level) and `name`.
        keep_uuid: If True, keeps the UUID of the old file,
            if that exists. Default is False when new UUID is generated.
        uuid: Set specific UUID for the file.
        date: Expected date in the input files. If not set,
            all files will be used. This might cause unexpected behavior if
            there are files from several days. If date is set as 'YYYY-MM-DD',
            only files that match the date will be used.

    Returns:
        2-element tuple containing

        - UUID of the generated file.
        - Files used in the processing.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    Examples:
        >>> from cloudnetpy.instruments import rpg2nc
        >>> site_meta = {'name': 'Hyytiala', 'altitude': 174}
        >>> rpg2nc('/path/to/files/', 'test.nc', site_meta)

    """
    l1_files = utils.get_sorted_filenames(path_to_l1_files, '.LV1')
    fmcw94_objects, valid_files = _get_fmcw94_objects(l1_files, date)
    one_day_of_data = create_one_day_data_record(fmcw94_objects)
    if not valid_files:
        return '', []
    fmcw = Fmcw94(one_day_of_data, site_meta)
    fmcw.convert_time_to_fraction_hour()
    fmcw.mask_invalid_ldr()
    fmcw.linear_to_db(('Zh', 'antenna_gain'))
    fmcw.add_site_meta()
    fmcw.add_zenith_angle()
    fmcw.add_height()
    attributes = output.add_time_attribute(RPG_ATTRIBUTES, fmcw.date)
    output.update_attributes(fmcw.data, attributes)
    return save_rpg(fmcw, output_file, valid_files, keep_uuid, uuid)


def create_one_day_data_record(rpg_objects: List[Union[Fmcw94Bin, HatproBin]]) -> dict:
    """Concatenates all RPG data from one day."""
    rpg_raw_data, rpg_header = _stack_rpg_data(rpg_objects)
    if len(rpg_objects) > 1:
        try:
            rpg_header = _reduce_header(rpg_header)
        except AssertionError as err:
            raise InconsistentDataError(f'{err}')
    rpg_raw_data = _mask_invalid_data(rpg_raw_data)
    return {**rpg_header, **rpg_raw_data}


def _stack_rpg_data(rpg_objects: List[Union[Fmcw94Bin, HatproBin]]) -> Tuple[dict, dict]:
    """Combines data from hourly RPG objects.

    Notes:
        Ignores variable names starting with an underscore.

    """
    def _stack(source, target, fun):
        for name, value in source.items():
            if not name.startswith('_'):
                target[name] = (fun((target[name], value)) if name in target else value)
    data, header = {}, {}
    for rpg in rpg_objects:
        _stack(rpg.data, data, np.concatenate)
        _stack(rpg.header, header, np.vstack)
    return data, header


def _reduce_header(header: dict) -> dict:
    """Removes duplicate header data."""
    header_out = header.copy()
    for name in header:
        first_row = header[name][0]
        assert np.isclose(header[name], first_row, rtol=1e-2).all(), f"Inconsistent header: {name}"
        header_out[name] = first_row
    return header_out


def _mask_invalid_data(data_in: dict) -> dict:
    """Masks zeros and other fill values from data."""
    data = data_in.copy()
    fill_values = (-999, 1e-10)
    for name in data:
        data[name] = ma.masked_equal(data[name], 0)
        for value in fill_values:
            data[name][data[name] == value] = ma.masked
            ind = np.isclose(data[name], value)
            data[name][ind] = ma.masked
    return data


def _get_fmcw94_objects(files: list, expected_date: Union[str, None]) -> Tuple[list, list]:
    """Creates a list of Rpg() objects from the file names."""
    objects = []
    valid_files = []
    for file in files:
        try:
            obj = Fmcw94Bin(file)
            if expected_date is not None:
                _validate_date(obj, expected_date)
        except (TypeError, ValueError) as err:
            logging.warning(err)
            continue
        objects.append(obj)
        valid_files.append(file)
    if objects:
        objects, valid_files = _remove_files_with_bad_height(objects, valid_files)
    if not valid_files:
        raise ValidTimeStampError
    return objects, valid_files


def _remove_files_with_bad_height(objects: list, files: list) -> Tuple[list, list]:
    lengths = [obj.data['Zh'].shape[1] for obj in objects]
    most_common = np.bincount(lengths).argmax()
    files = [file for file, obj, length in zip(files, objects, lengths) if length == most_common]
    objects = [obj for obj, length in zip(objects, lengths) if length == most_common]
    n_removed = len(lengths) - len(files)
    if n_removed > 0:
        logging.warning(f'Removed {n_removed} RPG-FMCW-94 files due to inconsistent height vector')
    return objects, files


def _validate_date(obj, expected_date: str) -> None:
    for t in obj.data['time'][:]:
        date_str = '-'.join(utils.seconds2date(t)[:3])
        if date_str != expected_date:
            raise ValueError('Ignoring a file (time stamps not what expected)')


class Rpg:
    """Base class for RPG FMCW-94 cloud radar and HATPRO mwr."""
    def __init__(self, raw_data: dict, site_meta: dict):
        self.raw_data = raw_data
        self.site_meta = site_meta
        self.date = self._get_date()
        self.location = site_meta['name']
        self.source = None
        self.data = {}

    def convert_time_to_fraction_hour(self) -> None:
        """Converts time to fraction hour."""
        key = 'time'
        fraction_hour = utils.seconds2hours(self.raw_data[key])
        self.raw_data[key] = fraction_hour
        self.data[key] = CloudnetArray(np.array(fraction_hour), key)

    def add_site_meta(self) -> None:
        for key, value in self.site_meta.items():
            if key in ('latitude', 'longitude', 'altitude'):
                self.data[key] = CloudnetArray(float(value), key)

    def _get_date(self) -> list:
        time_first = self.raw_data['time'][0]
        time_last = self.raw_data['time'][-1]
        date_first = utils.seconds2date(time_first)[:3]
        date_last = utils.seconds2date(time_last)[:3]
        if date_first != date_last:
            logging.warning('Measurements from different days')
        return date_first


class Fmcw94(Rpg):
    """Class for RPG FMCW-94 Cloud radar."""
    def __init__(self, raw_data: dict, site_properties: dict):
        super().__init__(raw_data, site_properties)
        self.data = self._init_data()
        self.source = 'RPG-FMCW-94'

    def add_zenith_angle(self) -> None:
        """Adds solar zenith angle."""
        elevation = self.data['elevation'].data
        zenith = 90 - elevation
        tolerance = 0.5
        difference = np.diff(zenith)
        if np.any(difference > tolerance):
            logging.warning(f'Varying zenith angle. Maximum difference: {max(difference)}')
        self.data['zenith_angle'] = CloudnetArray(zenith, 'zenith_angle')
        del self.data['elevation']

    def add_height(self):
        """Adds height vector."""
        zenith_angle = ma.median(self.data['zenith_angle'].data)
        height = utils.range_to_height(self.data['range'].data, float(zenith_angle))
        height += self.data['altitude'].data
        self.data['height'] = CloudnetArray(height, 'height')

    def linear_to_db(self, variables_to_log: tuple) -> None:
        """Changes linear units to logarithmic."""
        for name in variables_to_log:
            self.data[name].lin2db()

    def mask_invalid_ldr(self) -> None:
        """Removes ldr outliers."""
        threshold = -35
        if 'ldr' in self.data:
            self.data['ldr'].data = ma.masked_less_equal(self.data['ldr'].data, threshold)

    def _init_data(self) -> dict:
        data = {}
        for key in self.raw_data:
            data[key] = RadarArray(self.raw_data[key], key)
        return data


class Hatpro(Rpg):
    """Class for RPG HATPRO mwr."""
    def __init__(self, raw_data: dict, site_properties: dict):
        super().__init__(raw_data, site_properties)
        self.data = self._init_data()
        self.source = 'RPG-HATPRO'

    def sort_timestamps(self):
        key = 'LWP'
        if key not in self.data:
            logging.warning('No LWP vector, unable to sort')
        time = self.data['time'].data[:]
        array = self.data[key].data[:]
        ind = time.argsort()
        self.data['time'].data[:] = time[ind]
        self.data[key].data[:] = array[ind]

    def _init_data(self) -> dict:
        data = {}
        for key in self.raw_data:
            data[key] = CloudnetArray(self.raw_data[key], key)
        return data


def save_rpg(rpg: Rpg,
             output_file: str,
             valid_files: list,
             keep_uuid: bool,
             uuid: Union[str, None]) -> Tuple[str, list]:
    """Saves the RPG radar / mwr file."""
    dims = {'time': len(rpg.data['time'][:])}
    if 'fmcw' in rpg.source.lower():
        dims['range'] = len(rpg.data['range'][:])
        dims['chirp_sequence'] = len(rpg.data['chirp_start_indices'][:])
        file_type = 'radar'
    else:
        file_type = 'mwr'
    nc = output.init_file(output_file, dims, rpg.data, keep_uuid, uuid)
    file_uuid = nc.file_uuid
    output.write_common_level1b_parts(nc, rpg, file_type)
    nc.close()
    return file_uuid, valid_files


DEFINITIONS = {
    'model_number':
        ('\n'
         '0: Single polarisation radar.\n'
         '1: Dual polarisation radar.'),

    'dual_polarization':
        ('\n'
         'Value 0: Single polarisation radar.\n'
         'Value 1: Dual polarisation radar in linear depolarisation ratio (LDR)\n'
         '         mode.\n'
         'Value 2: Dual polarisation radar in simultaneous transmission\n'
         '         simultaneous reception (STSR) mode.'),

    'FFT_window':
        ('\n'
         'Value 0: Square\n'
         'Value 1: Parzen\n'
         'Value 2: Blackman\n'
         'Value 3: Welch\n'
         'Value 4: Slepian2\n'
         'Value 5: Slepian3'),

    'quality_flag':
        ('\n'
         'Bit 0: ADC saturation.\n'
         'Bit 1: Spectral width too high.\n'
         'Bit 2: No transmission power levelling.')

}

RPG_ATTRIBUTES = {
    # LDR-mode radars:
    'ldr': MetaData(
        long_name='Linear depolarisation ratio',
        units='dB'
    ),
    'rho_cx': MetaData(
        long_name='Co-cross-channel correlation coefficient',
        units='1'
    ),
    'phi_cx': MetaData(
        long_name='Co-cross-channel differential phase',
        units='rad'
    ),
    # STSR-mode radars
    'zdr': MetaData(
        long_name='Differential reflectivity',
        units='dB'
    ),
    'rho_hv': MetaData(
        long_name='Correlation coefficient',
        units='1'
    ),
    'phi_dp': MetaData(
        long_name='Differential phase',
        units='rad'
    ),
    'sldr': MetaData(
        long_name='Slanted linear depolarisation ratio',
        units='dB'
    ),
    'srho_hv': MetaData(
        long_name='Slanted correlation coefficient',
        units='1'
    ),
    'kdp': MetaData(
        long_name='Specific differential phase shift',
        units='rad km-1'
    ),
    'differential_attenuation': MetaData(
        long_name='',
        units='dB km-1'
    ),
    # All radars
    'file_code': MetaData(
        long_name='File code',
        units="1",
        comment='Indicates the RPG software version.',
    ),
    'program_number': MetaData(
        long_name='Program number',
        units="1"
    ),
    'model_number': MetaData(
        long_name='Model number',
        units="1",
        definition=DEFINITIONS['model_number']
    ),
    'antenna_separation': MetaData(
        long_name='Antenna separation',
        units='m',
    ),
    'antenna_diameter': MetaData(
        long_name='Antenna diameter',
        units='m',
    ),
    'antenna_gain': MetaData(
        long_name='Antenna gain',
        units='dB',
    ),
    'half_power_beam_width': MetaData(
        long_name='Half power beam width',
        units='degrees',
    ),
    'dual_polarization': MetaData(
        long_name='Dual polarisation type',
        units="1",
        definition=DEFINITIONS['dual_polarization']
    ),
    'sample_duration': MetaData(
        long_name='Sample duration',
        units='s'
    ),
    'calibration_interval': MetaData(
        long_name='Calibration interval in samples',
        units="1",
    ),
    'number_of_spectral_samples': MetaData(
        long_name='Number of spectral samples in each chirp sequence',
        units='1',
    ),
    'chirp_start_indices': MetaData(
        long_name='Chirp sequences start indices',
        units="1",
    ),
    'number_of_averaged_chirps': MetaData(
        long_name='Number of averaged chirps in sequence',
        units="1",
    ),
    'integration_time': MetaData(
        long_name='Integration time',
        units='s',
        comment='Effective integration time of chirp sequence',
    ),
    'range_resolution': MetaData(
        long_name='Vertical resolution of range',
        units='m',
    ),
    'FFT_window': MetaData(
        long_name='FFT window type',
        units="1",
        definition=DEFINITIONS['FFT_window']
    ),
    'input_voltage_range': MetaData(
        long_name='ADC input voltage range (+/-)',
        units='mV',
    ),
    'noise_threshold': MetaData(
        long_name='Noise filter threshold factor',
        units='1',
        comment='Multiple of the standard deviation of Doppler spectra.'
    ),
    'time_ms': MetaData(
        long_name='Time ms',
        units='ms',
    ),
    'quality_flag': MetaData(
        long_name='Quality flag',
        definition=DEFINITIONS['quality_flag'],
        units="1",
    ),
    'voltage': MetaData(
        long_name='Voltage',
        units='V',
    ),
    'brightness_temperature': MetaData(
        long_name='Brightness temperature',
        units='K',
    ),
    'if_power': MetaData(
        long_name='IF power at ACD',
        units='uW',
    ),
    'azimuth_angle': MetaData(
        long_name='Azimuth angle',
        standard_name='solar_azimuth_angle',
        units='degree',
    ),
    'status_flag': MetaData(
        long_name='Status flag for heater and blower',
        units="1",
    ),
    'transmitted_power': MetaData(
        long_name='Transmitted power',
        units='W',
    ),
    'transmitter_temperature': MetaData(
        long_name='Transmitter temperature',
        units='K',
    ),
    'receiver_temperature': MetaData(
        long_name='Receiver temperature',
        units='K',
    ),
    'pc_temperature': MetaData(
        long_name='PC temperature',
        units='K',
    ),
    'skewness': MetaData(
        long_name='Skewness of spectra',
        units='1',
    ),
    'kurtosis': MetaData(
        long_name='Kurtosis of spectra',
        units='1',
    ),
    'correlation_coefficient': MetaData(
        long_name='Correlation coefficient',
        units="1",
    ),
    'wind_direction': MetaData(
        long_name='Wind direction',
        units='degrees',
    ),
    'wind_speed': MetaData(
        long_name='Wind speed',
        units='m s-1',
    ),
    'relative_humidity': MetaData(
        long_name='Relative humidity',
        units='%',
    ),
    'lwp': MetaData(
        units='g m-2'
    )
}
