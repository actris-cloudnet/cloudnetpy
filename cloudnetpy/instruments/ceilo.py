"""Module for reading and processing Vaisala / Lufft ceilometers."""
import linecache
from typing import Union, Optional
import numpy as np
from cloudnetpy.instruments.lufft import LufftCeilo
from cloudnetpy.instruments.vaisala import ClCeilo, Ct25k
from cloudnetpy import utils, output, CloudnetArray
from cloudnetpy.metadata import MetaData


def ceilo2nc(full_path: str,
             output_file: str,
             site_meta: dict,
             keep_uuid: Optional[bool] = False,
             uuid: Optional[str] = None,
             date: Optional[str] = None) -> str:
    """Converts Vaisala / Lufft ceilometer data into Cloudnet Level 1b netCDF file.

    This function reads raw Vaisala (CT25k, CL31, CL51) and Lufft (CHM15k)
    ceilometer files and writes the data into netCDF file. Three variants
    of the attenuated backscatter are saved in the file:

        1. Raw backscatter, `beta_raw`
        2. Signal-to-noise screened backscatter, `beta`
        3. SNR-screened backscatter with smoothed weak background, `beta_smooth`

    Args:
        full_path: Ceilometer file name. For Vaisala it is a text file, for CHM15k it is
            a netCDF file.
        output_file: Output file name, e.g. 'ceilo.nc'.
        site_meta: Dictionary containing information about the site and instrument.
            Required key value pairs are `name` and `altitude` (metres above mean sea level).
            Also 'calibration_factor' is recommended because the default value is probably
            incorrect.
        keep_uuid: If True, keeps the UUID of the old file, if that exists. Default is False
            when new UUID is generated.
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        RuntimeError: Failed to read or process raw ceilometer data.

    Examples:
        >>> from cloudnetpy.instruments import ceilo2nc
        >>> site_meta = {'name': 'Mace-Head', 'altitude': 5}
        >>> ceilo2nc('vaisala_raw.txt', 'vaisala.nc', site_meta)
        >>> site_meta = {'name': 'Juelich', 'altitude': 108, 'calibration_factor': 2.3e-12}
        >>> ceilo2nc('chm15k_raw.nc', 'chm15k.nc', site_meta)

    """
    ceilo = _initialize_ceilo(full_path, date)
    ceilo.read_ceilometer_file(site_meta.get('calibration_factor', None))
    beta_variants = ceilo.calc_beta()
    _append_data(ceilo, beta_variants)
    _append_height(ceilo, site_meta['altitude'])
    attributes = output.add_time_attribute(ATTRIBUTES, ceilo.date)
    output.update_attributes(ceilo.data, attributes)
    return _save_ceilo(ceilo, output_file, site_meta['name'], keep_uuid, uuid)


def _initialize_ceilo(full_path: str,
                      date: Union[str, None]) -> Union[ClCeilo, Ct25k, LufftCeilo]:
    model = _find_ceilo_model(full_path)
    if model == 'cl31_or_cl51':
        return ClCeilo(full_path, date)
    if model == 'ct25k':
        return Ct25k(full_path)
    return LufftCeilo(full_path, date)


def _find_ceilo_model(full_path: str) -> str:
    if full_path.lower().endswith('.nc'):
        return 'chm15k'
    first_empty_line = utils.find_first_empty_line(full_path)
    max_number_of_empty_lines = 10
    for n in range(1, max_number_of_empty_lines):
        line = linecache.getline(full_path, first_empty_line + n)
        if not utils.is_empty_line(line):
            line = linecache.getline(full_path, first_empty_line + n + 1)
            break
    if 'CL' in line:
        return 'cl31_or_cl51'
    if 'CT' in line:
        return 'ct25k'
    raise RuntimeError('Error: Unknown ceilo model.')


def _append_height(ceilo: Union[ClCeilo, Ct25k, LufftCeilo],
                   site_altitude: float) -> None:
    """Finds height above mean sea level."""
    tilt_angle = np.median(ceilo.metadata['tilt_angle'])
    height = utils.range_to_height(ceilo.range, float(tilt_angle))
    height += float(site_altitude)
    ceilo.data['height'] = CloudnetArray(np.array(height), 'height')
    ceilo.data['altitude'] = CloudnetArray(site_altitude, 'altitude')


def _append_data(ceilo: Union[ClCeilo, Ct25k, LufftCeilo],
                 beta_variants: tuple):
    """Adds data / metadata as CloudnetArrays to ceilo.data."""
    for data, name in zip(beta_variants, ('beta_raw', 'beta', 'beta_smooth')):
        ceilo.data[name] = CloudnetArray(data, name)
    for field in ('range', 'time', 'wavelength', 'calibration_factor'):
        ceilo.data[field] = CloudnetArray(np.array(getattr(ceilo, field)), field)
    for field, data in ceilo.metadata.items():
        first_element = data if utils.isscalar(data) else data[0]
        if not isinstance(first_element, str):  # String array writing not yet supported
            ceilo.data[field] = CloudnetArray(np.array(ceilo.metadata[field], dtype=float), field)


def _save_ceilo(ceilo: Union[ClCeilo, Ct25k, LufftCeilo],
                output_file: str,
                location: str,
                keep_uuid: bool,
                uuid: Union[str, None]) -> str:
    """Saves the ceilometer netcdf-file."""

    dims = {'time': len(ceilo.time),
            'range': len(ceilo.range)}

    rootgrp = output.init_file(output_file, dims, ceilo.data, keep_uuid, uuid)
    uuid = rootgrp.file_uuid
    output.add_file_type(rootgrp, 'lidar')
    rootgrp.title = f"Ceilometer file from {location}"
    rootgrp.year, rootgrp.month, rootgrp.day = ceilo.date
    rootgrp.location = location
    rootgrp.history = f"{utils.get_time()} - ceilometer file created"
    rootgrp.source = ceilo.model
    output.add_references(rootgrp)
    rootgrp.close()
    return uuid


ATTRIBUTES = {
    'beta': MetaData(
        long_name='Attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment='Range corrected, SNR screened, attenuated backscatter.'
    ),
    'beta_raw': MetaData(
        long_name='Raw attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment="Range corrected, attenuated backscatter."
    ),
    'beta_smooth': MetaData(
        long_name='Smoothed attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment=('Range corrected, SNR screened backscatter coefficient.\n'
                 'Weak background is smoothed using Gaussian 2D-kernel.')
    ),
    'scale': MetaData(
        long_name='Scale',
        units='%',
        comment='100 (%) is normal.'
    ),
    'software_level': MetaData(
        long_name='Software level ID',
        units='',
    ),
    'laser_temperature': MetaData(
        long_name='Laser temperature',
        units='C',
    ),
    'window_transmission': MetaData(
        long_name='Window transmission estimate',
        units='%',
    ),
    'tilt_angle': MetaData(
        long_name='Tilt angle from vertical',
        units='degrees',
    ),
    'laser_energy': MetaData(
        long_name='Laser pulse energy',
        units='%',
    ),
    'background_light': MetaData(
        long_name='Background light',
        units='mV',
        comment='Measured at internal ADC input.'
    ),
    'backscatter_sum': MetaData(
        long_name='Sum of detected and normalized backscatter',
        units='sr-1',
        comment='Multiplied by scaling factor times 1e4.',
    ),
    'range_resolution': MetaData(
        long_name='Range resolution',
        units='m',
    ),
    'number_of_gates': MetaData(
        long_name='Number of range gates in profile',
        units='',
    ),
    'unit_id': MetaData(
        long_name='Ceilometer unit number',
        units='',
    ),
    'message_number': MetaData(
        long_name='Message number',
        units='',
    ),
    'message_subclass': MetaData(
        long_name='Message subclass number',
        units='',
    ),
    'detection_status': MetaData(
        long_name='Detection status',
        units='',
        comment='From the internal software of the instrument.'
    ),
    'warning': MetaData(
        long_name='Warning and Alarm flag',
        units='',
        definition=('\n'
                    'Value 0: Self-check OK\n'
                    'Value W: At least one warning on\n'
                    'Value A: At least one error active.')
    ),
    'warning_flags': MetaData(
        long_name='Warning flags',
        units='',
    ),
    'receiver_sensitivity': MetaData(
        long_name='Receiver sensitivity',
        units='%',
        comment='Expressed as % of nominal factory setting.'
    ),
    'window_contamination': MetaData(
        long_name='Window contamination',
        units='mV',
        comment='Measured at internal ADC input.'
    ),
    'calibration_factor': MetaData(
        long_name='Backscatter calibration factor',
        units='',
    ),
    'wavelength': MetaData(
        long_name='Laser wavelength',
        units='nm',
    )
}
