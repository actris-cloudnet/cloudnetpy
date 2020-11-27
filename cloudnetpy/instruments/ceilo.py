"""Module for reading and processing Vaisala / Jenoptik ceilometers."""
import linecache
from typing import Union
import numpy as np
from cloudnetpy.instruments.jenoptik import JenoptikCeilo
from cloudnetpy.instruments.vaisala import Cl31, Cl51, Ct25k
from cloudnetpy import utils, output, CloudnetArray
from cloudnetpy.metadata import MetaData


def ceilo2nc(input_file: str,
             output_file: str,
             site_meta: dict,
             keep_uuid: bool = False,
             uuid: Union[str, None] = None) -> str:
    """Converts Vaisala and Jenoptik raw files into netCDF file.

    This function reads raw Vaisala (CT25k, CL31, CL51) and Jenoptik (CHM15k)
    ceilometer files and writes the data into netCDF file. Three variants
    of the attenuated backscatter are saved in the file:

        1. Raw backscatter, `beta_raw`
        2. Signal-to-noise screened backscatter, `beta`
        3. SNR-screened backscatter with smoothed weak background, `beta_smooth`

    Args:
        input_file (str): Ceilometer file name. For Vaisala it is a text file,
            for Jenoptik it is a netCDF file.
        output_file (str): Output file name, e.g. 'ceilo.nc'.
        site_meta (dict): Dictionary containing information about the
            site. Required key value pairs are `name` and `altitude`
            (metres above mean sea level).
        keep_uuid (bool, optional): If True, keeps the UUID of the old file,
            if that exists. Default is False when new UUID is generated.
        uuid (str, optional): Set specific UUID for the file.

    Returns:
        str: UUID of the generated file.

    Raises:
        RuntimeError: Failed to read or process raw ceilometer data.

    Examples:
        >>> from cloudnetpy.instruments import ceilo2nc
        >>> site_meta = {'name': 'Mace-Head', 'altitude':5}
        >>> ceilo2nc('vaisala_raw.txt', 'vaisala.nc', site_meta)
        >>> ceilo2nc('jenoptik_raw.nc', 'jenoptik.nc', site_meta)

    """
    ceilo = _initialize_ceilo(input_file, site_meta['name'])
    ceilo.read_ceilometer_file()
    beta_variants = ceilo.calc_beta()
    _append_data(ceilo, beta_variants)
    _append_height(ceilo, site_meta['altitude'])
    output.update_attributes(ceilo.data, ATTRIBUTES)
    return _save_ceilo(ceilo, output_file, site_meta['name'], keep_uuid, uuid)


def _initialize_ceilo(file, site_name):
    model = _find_ceilo_model(file)
    if model == 'cl51':
        return Cl51(file)
    elif model == 'cl31':
        return Cl31(file)
    elif model == 'ct25k':
        return Ct25k(file)
    elif model == 'chm15k':
        return JenoptikCeilo(file, site_name)
    raise RuntimeError('Error: Unknown ceilo model.')


def _find_ceilo_model(file):
    if file.endswith('nc'):
        return 'chm15k'
    first_empty_line = utils.find_first_empty_line(file)
    hint = linecache.getline(file, first_empty_line + 2)[1:5]
    if hint == 'CL01':
        return 'cl51'
    elif hint == 'CL02':
        return 'cl31'
    elif hint == 'CT02':
        return 'ct25k'
    return None


def _append_height(ceilo, site_altitude):
    """Finds height above mean sea level."""
    tilt_angle = np.median(ceilo.metadata['tilt_angle'])
    height = utils.range_to_height(ceilo.range, tilt_angle)
    height += float(site_altitude)
    ceilo.data['height'] = CloudnetArray(height, 'height')


def _append_data(ceilo, beta_variants):
    """Add data and metadata as CloudnetArray's to ceilo.data attribute."""
    for data, name in zip(beta_variants, ('beta_raw', 'beta', 'beta_smooth')):
        ceilo.data[name] = CloudnetArray(data, name)
    for field in ('range', 'time'):
        ceilo.data[field] = CloudnetArray(getattr(ceilo, field), field)
    for field, data in ceilo.metadata.items():
        first_element = data if utils.isscalar(data) else data[0]
        if not isinstance(first_element, str):  # String array writing not yet supported
            ceilo.data[field] = CloudnetArray(np.array(ceilo.metadata[field],
                                                       dtype=float), field)
    if hasattr(ceilo, 'wavelength'):
        ceilo.data['wavelength'] = CloudnetArray(ceilo.wavelength, 'wavelength', 'nm')


def _save_ceilo(ceilo, output_file, location, keep_uuid, uuid: Union[str, None] = None) -> str:
    """Saves the ceilometer netcdf-file."""
    dims = {'time': len(ceilo.time),
            'range': len(ceilo.range)}
    rootgrp = output.init_file(output_file, dims, ceilo.data, keep_uuid, uuid)
    uuid = rootgrp.file_uuid
    output.add_file_type(rootgrp, 'lidar')
    if hasattr(ceilo, 'dataset'):
        output.copy_variables(ceilo.dataset, rootgrp, ('wavelength',))
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
    )
}
