"""Module that generates Cloudnet categorize file."""
from typing import Union, Optional
from cloudnetpy import output, utils
from cloudnetpy.categorize import atmos, classify
from cloudnetpy.metadata import MetaData
from cloudnetpy.categorize.radar import Radar
from cloudnetpy.categorize.model import Model
from cloudnetpy.categorize.mwr import Mwr
from cloudnetpy.categorize.lidar import Lidar


def generate_categorize(input_files: dict,
                        output_file: str,
                        uuid: Optional[str] = None) -> str:
    """Generates Cloudnet Level 1c categorize file.

    The measurements are rebinned into a common height / time grid,
    and classified as different types of scatterers such as ice, liquid,
    insects, etc. Next, the radar signal is corrected for atmospheric
    attenuation, and error estimates are computed. Results are saved
    in *ouput_file* which is a compressed netCDF4 file.

    Args:
        input_files: dict containing file names for calibrated `radar`, `lidar`, `model` and
            `mwr` files.
        output_file: Full path of the output file.
        uuid: Set specific UUID for the file.

    Returns:
        UUID of the generated file.

    Raises:
        RuntimeError: Failed to create the categorize file.

    Notes:
        Separate mwr-file is not needed when using RPG cloud radar which
        measures liquid water path. Then, the radar file can be used as
        a mwr-file as well, i.e. {'mwr': 'radar.nc'}.

    Examples:
        >>> from cloudnetpy.categorize import generate_categorize
        >>> input_files = {'radar': 'radar.nc',
                           'lidar': 'lidar.nc',
                           'model': 'model.nc',
                           'mwr': 'mwr.nc'}
        >>> generate_categorize(input_files, 'output.nc')

    """

    def _interpolate_to_cloudnet_grid() -> list:
        wl_band = utils.get_wl_band(data['radar'].radar_frequency)
        data['model'].interpolate_to_common_height(wl_band)
        data['model'].interpolate_to_grid(time, height)
        data['mwr'].rebin_to_grid(time)
        radar_data_gap_indices = data['radar'].rebin_to_grid(time)
        lidar_data_gap_indices = data['lidar'].interpolate_to_grid(time, height)
        bad_time_indices = list(set(radar_data_gap_indices + lidar_data_gap_indices))
        valid_ind = [ind for ind in range(len(time)) if ind not in bad_time_indices]
        return valid_ind

    def _screen_bad_time_indices(valid_indices: list) -> None:
        n_time_full = len(time)
        data['radar'].time = time[valid_indices]
        for var in ('radar', 'lidar', 'mwr', 'model'):
            for key, item in data[var].data.items():
                if utils.isscalar(item.data):
                    continue
                array = item[:]
                if array.shape[0] == n_time_full:
                    if array.ndim == 1:
                        array = array[valid_indices]
                    elif array.ndim == 2:
                        array = array[valid_indices, :]
                    else:
                        continue
                    data[var].data[key].data = array
        for key, item in data['model'].data_dense.items():
            data['model'].data_dense[key] = item[valid_indices, :]

    def _prepare_output() -> dict:
        data['radar'].add_meta()
        data['model'].screen_sparse_fields()
        for key in ('category_bits', 'rain_rate'):
            data['radar'].append_data(getattr(classification, key), key)
        for key in ('radar_liquid_atten', 'radar_gas_atten'):
            data['radar'].append_data(attenuations[key], key)
        data['radar'].append_data(quality['quality_bits'], 'quality_bits')
        return {**data['radar'].data,
                **data['lidar'].data,
                **data['model'].data,
                **data['model'].data_sparse,
                **data['mwr'].data}

    def _define_dense_grid():
        return utils.time_grid(), data['radar'].height

    def _close_all():
        for obj in data.values():
            obj.close()

    data = {
        'radar': Radar(input_files['radar']),
        'lidar': Lidar(input_files['lidar']),
        'mwr': Mwr(input_files['mwr'])
    }
    data['model'] = Model(input_files['model'], data['radar'].altitude)
    time, height = _define_dense_grid()
    valid_ind = _interpolate_to_cloudnet_grid()
    _screen_bad_time_indices(valid_ind)
    if 'rpg' in data['radar'].type.lower() or 'basta' in data['radar'].type.lower():
        data['radar'].filter_speckle_noise()
        data['radar'].filter_1st_gate_artifact()
    for variable in ('v', 'v_sigma', 'ldr'):
        data['radar'].filter_stripes(variable)
    data['radar'].remove_incomplete_pixels()
    data['model'].calc_wet_bulb()
    classification = classify.classify_measurements(data)
    attenuations = atmos.get_attenuations(data, classification)
    data['radar'].correct_atten(attenuations)
    data['radar'].calc_errors(attenuations, classification)
    quality = classify.fetch_quality(data, classification, attenuations)
    cloudnet_arrays = _prepare_output()
    date = data['radar'].get_date()
    attributes = output.add_time_attribute(CATEGORIZE_ATTRIBUTES, date)
    attributes = output.add_time_attribute(attributes, date, 'model_time')
    attributes = output.add_source_attribute(attributes, data)
    output.update_attributes(cloudnet_arrays, attributes)
    uuid = _save_cat(output_file, data, cloudnet_arrays, uuid)
    _close_all()
    return uuid


def _save_cat(full_path: str,
              data_obs: dict,
              cloudnet_arrays: dict,
              uuid: Union[str, None]) -> str:
    """Creates a categorize netCDF4 file and saves all data into it."""

    dims = {'time': len(data_obs['radar'].time),
            'height': len(data_obs['radar'].height),
            'model_time': len(data_obs['model'].time),
            'model_height': len(data_obs['model'].mean_height)}

    file_type = 'categorize'
    nc = output.init_file(full_path, dims, cloudnet_arrays, uuid)
    uuid = nc.file_uuid
    nc.cloudnet_file_type = file_type
    output.copy_global(data_obs['radar'].dataset, nc, ('year', 'month', 'day', 'location'))
    nc.title = f"Cloud categorization products from {data_obs['radar'].location}"
    nc.source_file_uuids = output.get_source_uuids(*data_obs.values())
    nc.references = output.get_references(file_type)
    output.add_source_instruments(nc, data_obs)
    output.merge_history(nc, file_type, data_obs)
    nc.close()
    return uuid


COMMENTS = {
    'category_bits':
    ('This variable contains information on the nature of the targets\n'
     'at each pixel, thereby facilitating the application of algorithms that work\n'
     'with only one type of target. The information is in the form of an array of\n'
     'bits, each of which states either whether a certain type of particle is present\n'
     '(e.g. aerosols), or the whether some of the target particles have a particular\n'
     'property. The definitions of each bit are given in the definition attribute.\n'
     'Bit 0 is the least significant.'),

    'quality_bits':
    ('This variable contains information on the quality of the data at each pixel.\n'
     'The information is in the form of an array of bits, and the definitions of each bit\n'
     'are given in the definition attribute. Bit 0 is the least significant.'),

    'lwp':
    ('This variable is the vertically integrated liquid water directly over the site.\n'
     'The temporal correlation of errors in liquid water path means that it is not really\n'
     'meaningful to distinguish bias from random error, so only an error variable is provided.'),

    'radar_liquid_atten':
    ('This variable was calculated from the liquid water path measured by microwave radiometer\n'
     'using lidar and radar returns to perform an approximate partitioning of the liquid water\n'
     'content with height. Bit 5 of the quality_bits variable indicates where a correction for\n'
     'liquid water attenuation has been performed.'),

    'radar_gas_atten':
    ('This variable was calculated from the model temperature, pressure and humidity, but forcing\n'
     'pixels containing liquid cloud to saturation with respect to liquid water.\n'
     'It has been used to correct Z.'),

    'Tw': 'This variable was derived from model temperature, pressure and relative humidity.',

    'Z_sensitivity':
    ('This variable is an estimate of the radar sensitivity, i.e. the minimum detectable radar\n'
     'reflectivity, as a function of height. It includes the effect of ground clutter and gas\n'
     'attenuation but not liquid attenuation.'),

    'Z_error':
    ('This variable is an estimate of the one-standard-deviation random error in radar\n'
     'reflectivity factor. It originates from the following independent sources of error:\n'
     '1) Precision in reflectivity estimate due to finite signal to noise and finite number\n'
     '   of pulses\n'
     '2) 10% uncertainty in gaseous attenuation correction (mainly due to error in model\n'
     '   humidity field)\n'
     '3) Error in liquid water path (given by the variable lwp_error) and its partitioning\n'
     '   with height).'),

    'Z':
    ('This variable has been corrected for attenuation by gaseous attenuation (using the\n'
     'thermodynamic variables from a forecast model; see the radar_gas_atten variable) and\n'
     'liquid attenuation (using liquid water path from a microwave radiometer; see the\n'
     'radar_liquid_atten variable) but rain and melting-layer attenuation has not been corrected.\n'
     'Calibration convention: in the absence of attenuation, a cloud at 273 K containing\n'
     'one million 100-micron droplets per cubic metre will have a reflectivity of 0 dBZ\n'
     'at all frequencies.'),

    'bias': 'This variable is an estimate of the one-standard-deviation calibration error.',

}

DEFINITIONS = {
    'category_bits':
        ('\n'
         'Bit 0: Small liquid droplets are present.\n'
         'Bit 1: Falling hydrometeors are present; if Bit 2 is set then these are most likely\n'
         '       ice particles, otherwise they are drizzle or rain drops.\n'
         'Bit 2: Wet-bulb temperature is less than 0 degrees C, implying the phase of\n'
         '       Bit-1 particles.\n'
         'Bit 3: Melting ice particles are present.\n'
         'Bit 4: Aerosol particles are present and visible to the lidar.\n'
         'Bit 5: Insects are present and visible to the radar.'),

    'quality_bits':
        ('\n'
         'Bit 0: An echo is detected by the radar.\n'
         'Bit 1: An echo is detected by the lidar.\n'
         'Bit 2: The apparent echo detected by the radar is ground clutter or some other\n'
         '       non-atmospheric artifact.\n'
         'Bit 3: The lidar echo is due to clear-air molecular scattering.\n'
         'Bit 4: Liquid water cloud, rainfall or melting ice below this pixel\n'
         '       will have caused radar and lidar attenuation; if bit 5 is set then\n'
         '       a correction for the radar attenuation has been performed;\n'
         '       otherwise do not trust the absolute values of reflectivity factor.\n'
         '       No correction is performed for lidar attenuation.\n'
         'Bit 5: Radar reflectivity has been corrected for liquid-water attenuation\n'
         '       using the microwave radiometer measurements of liquid water path\n'
         '       and the lidar estimation of the location of liquid water cloud;\n'
         '       be aware that errors in reflectivity may result.\n')
}

CATEGORIZE_ATTRIBUTES = {
    # Radar variables
    'Z': MetaData(
        long_name='Radar reflectivity factor',
        units='dBZ',
        comment=COMMENTS['Z'],
        ancillary_variables='Z_error Z_bias Z_sensitivity'
    ),
    'Z_error': MetaData(
        long_name='Error in radar reflectivity factor',
        units='dB',
        comment=COMMENTS['Z_error']
    ),
    'Z_bias': MetaData(
        long_name='Bias in radar reflectivity factor',
        units='dB',
        comment=COMMENTS['bias']
    ),
    'Z_sensitivity': MetaData(
        long_name='Minimum detectable radar reflectivity',
        units='dBZ',
        comment=COMMENTS['Z_sensitivity']
    ),
    'v_sigma': MetaData(
        long_name='Standard deviation of mean Doppler velocity',
        units='m s-1'
    ),
    # Lidar variables
    'beta': MetaData(
        long_name='Attenuated backscatter coefficient',
        units='sr-1 m-1',
        comment='SNR-screened attenuated backscatter coefficient.',
        ancillary_variables='beta_error beta_bias'
    ),
    'beta_error': MetaData(
        long_name='Error in attenuated backscatter coefficient',
        units='dB',
    ),
    'beta_bias': MetaData(
        long_name='Bias in attenuated backscatter coefficient',
        units='dB',
    ),
    'lidar_wavelength': MetaData(
        long_name='Laser wavelength',
        units='nm'
    ),
    # MWR variables
    'lwp_error': MetaData(
        long_name='Error in liquid water path',
        units='g m-2',
    ),
    'lwp': MetaData(
        ancillary_variables='lwp_error'
    ),
    # Model variables
    'Tw': MetaData(
        long_name='Wet-bulb temperature',
        units='K',
        comment=COMMENTS['Tw']
    ),
    'model_time': MetaData(
        long_name='Model time UTC',
        calendar='standard'
    ),
    'model_height': MetaData(
        long_name='Height of model variables above mean sea level',
        units='m',
        axis='Z'
    ),
    'vwind': MetaData(
        long_name='Meridional wind',
        units='m s-1',
    ),
    'uwind': MetaData(
        long_name='Zonal wind',
        units='m s-1',
    ),
    'q': MetaData(
        long_name='Specific humidity',
        units='1'
    ),
    # MISC
    'category_bits': MetaData(
        long_name='Target categorization bits',
        comment=COMMENTS['category_bits'],
        definition=DEFINITIONS['category_bits'],
        units='1'
    ),
    'quality_bits': MetaData(
        long_name='Data quality bits',
        comment=COMMENTS['quality_bits'],
        definition=DEFINITIONS['quality_bits'],
        units='1'
    ),
    'rain_rate': MetaData(
        long_name='Rain rate',
        units='mm h-1',
        comment='Fill values denote rain with undefined intensity.'
    ),
    'radar_liquid_atten': MetaData(
        long_name='Two-way radar attenuation due to liquid water',
        units='dB',
        comment=COMMENTS['radar_liquid_atten']
    ),
    'radar_gas_atten': MetaData(
        long_name='Two-way radar attenuation due to atmospheric gases',
        units='dB',
        comment=COMMENTS['radar_gas_atten'],
        references='Liebe (1985, Radio Sci. 20(5), 1069-1089)'
    ),
}
