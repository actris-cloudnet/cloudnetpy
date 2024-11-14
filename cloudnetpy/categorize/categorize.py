"""Module that generates Cloudnet categorize file."""

import dataclasses
import logging
from dataclasses import fields

import numpy as np
from numpy.typing import NDArray

from cloudnetpy import output, utils
from cloudnetpy.categorize import attenuation, classify
from cloudnetpy.categorize.containers import Observations
from cloudnetpy.categorize.disdrometer import Disdrometer
from cloudnetpy.categorize.lidar import Lidar
from cloudnetpy.categorize.model import Model
from cloudnetpy.categorize.mwr import Mwr
from cloudnetpy.categorize.radar import Radar
from cloudnetpy.datasource import DataSource
from cloudnetpy.exceptions import DisdrometerDataError, ValidTimeStampError
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import CategoryBits, QualityBits


def generate_categorize(
    input_files: dict,
    output_file: str,
    uuid: str | None = None,
    options: dict | None = None,
) -> str:
    """Generates a Cloudnet Level 1c categorize file.

    This function rebins measurements into a common height/time grid
    and classifies them into different scatterer types, such as ice,
    liquid, insects, etc. The radar signal is corrected for atmospheric
    attenuation, and error estimates are computed. The results are saved in
    *output_file*, a compressed netCDF4 file.

    Args:
        input_files (dict): Contains filenames for calibrated `radar`, `lidar`,
            and `model` files. Optionally, it can also include `disdrometer`,
            `mwr` (containing the LWP variable), and `lv0_files` (a list of RPG
            Level 0 files).
        output_file (str): The full path of the output file.
        uuid (str): Specific UUID to assign to the generated file.
        options (dict): Dictionary containing optional parameters.

    Returns:
        str: UUID of the generated file.

    Raises:
        RuntimeError: Raised if the categorize file creation fails.

    Notes:
        A separate MWR file is not required when using an RPG cloud radar that
        measures liquid water path (LWP). In this case, the radar file can also
        serve as the MWR file (e.g., {'mwr': 'radar.nc'}). If no MWR file
        is provided, liquid attenuation correction cannot be performed.

        If RPG L0 files are included as additional input, the Voodoo method
        is used to detect liquid droplets.

    Examples:
        >>> from cloudnetpy.categorize import generate_categorize
        >>> input_files = {
        ...     'radar': 'radar.nc',
        ...     'lidar': 'lidar.nc',
        ...     'model': 'model.nc',
        ...     'mwr': 'mwr.nc'
        ... }
        >>> generate_categorize(input_files, 'output.nc')

        >>> input_files['lv0_files'] = ['file1.LV0', 'file2.LV0']  # Add RPG LV0 files
        >>> generate_categorize(input_files, 'output.nc')  # Use the Voodoo method
    """

    def _interpolate_to_cloudnet_grid() -> list[int]:
        if data.disdrometer is not None:
            data.disdrometer.interpolate_to_grid(time)
        if data.mwr is not None:
            data.mwr.rebin_to_grid(time)
        data.model.calc_attenuations(data.radar.radar_frequency)
        data.model.interpolate_to_common_height()
        model_gap_ind = data.model.interpolate_to_grid(time, height)
        radar_gap_ind = data.radar.rebin_to_grid(time)
        lidar_gap_ind = data.lidar.interpolate_to_grid(time, height)
        gap_indices = set(radar_gap_ind + lidar_gap_ind + model_gap_ind)
        return [ind for ind in range(len(time)) if ind not in gap_indices]

    def _screen_bad_time_indices(valid_indices: list) -> None:
        n_time_full = len(time)
        data.radar.time = time[valid_indices]
        for field in fields(data):
            obj = getattr(data, field.name)
            if not hasattr(obj, "data"):
                continue
            for key, item in obj.data.items():
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
                    obj.data[key].data = array
        for key, item in data.model.data_dense.items():
            data.model.data_dense[key] = item[valid_indices, :]

    def _prepare_output() -> dict:
        data.radar.add_meta()
        data.model.screen_sparse_fields()

        if data.disdrometer is not None:
            data.radar.data.pop("rainfall_rate", None)
            data.disdrometer.data.pop("n_particles", None)

        data.radar.append_data(attenuations.gas.amount, "radar_gas_atten")
        data.radar.append_data(attenuations.liquid.amount, "radar_liquid_atten")
        data.radar.append_data(attenuations.rain.amount, "radar_rain_atten")
        data.radar.append_data(attenuations.melting.amount, "radar_melting_atten")

        data.radar.append_data(_classes_to_bits(quality), "quality_bits")

        data.radar.append_data(classification.insect_prob, "insect_prob")
        data.radar.append_data(classification.is_rain, "rain_detected")
        data.radar.append_data(
            _classes_to_bits(classification.category_bits), "category_bits"
        )

        if classification.liquid_prob is not None:
            data.radar.append_data(classification.liquid_prob, "liquid_prob")

        return {
            **data.radar.data,
            **data.lidar.data,
            **data.model.data,
            **data.model.data_sparse,
            **(data.mwr.data if data.mwr is not None else {}),
            **(data.disdrometer.data if data.disdrometer is not None else {}),
        }

    def _define_dense_grid() -> tuple:
        return utils.time_grid(), data.radar.height

    def _close_all() -> None:
        for field in fields(data):
            obj = getattr(data, field.name)
            if isinstance(obj, DataSource):
                obj.close()

    try:
        radar = Radar(input_files["radar"])
        data = Observations(
            radar=radar,
            lidar=Lidar(input_files["lidar"]),
            model=Model(input_files["model"], radar.altitude, options),
            lv0_files=input_files.get("lv0_files"),
        )
        if "mwr" in input_files:
            data.mwr = Mwr(input_files["mwr"])
        if "disdrometer" in input_files:
            try:
                data.disdrometer = Disdrometer(input_files["disdrometer"])
            except DisdrometerDataError as err:
                logging.warning("Unable to use disdrometer: %s", err)
        time, height = _define_dense_grid()
        valid_ind = _interpolate_to_cloudnet_grid()
        if not valid_ind:
            msg = "No overlapping radar and lidar timestamps found"
            raise ValidTimeStampError(msg)
        _screen_bad_time_indices(valid_ind)
        if any(source in data.radar.source_type.lower() for source in ("rpg", "basta")):
            data.radar.filter_speckle_noise()
            data.radar.filter_1st_gate_artifact()
        for variable in ("v", "v_sigma", "ldr"):
            data.radar.filter_stripes(variable)
        data.radar.remove_incomplete_pixels()
        data.model.calc_wet_bulb()

        classification = classify.classify_measurements(data)

        attenuations = attenuation.get_attenuations(data, classification)

        data.radar.correct_atten(attenuations)
        data.radar.calc_errors(attenuations, classification.is_clutter)

        quality = classify.fetch_quality(data, classification, attenuations)

        cloudnet_arrays = _prepare_output()

        date = data.radar.get_date()
        attributes = output.add_time_attribute(CATEGORIZE_ATTRIBUTES, date)
        attributes = output.add_time_attribute(attributes, date, "model_time")
        attributes = output.add_source_attribute(attributes, data)
        output.update_attributes(cloudnet_arrays, attributes)
        return _save_cat(output_file, data, cloudnet_arrays, uuid)
    finally:
        _close_all()


def _save_cat(
    full_path: str,
    data_obs: Observations,
    cloudnet_arrays: dict,
    uuid: str | None,
) -> str:
    """Creates a categorize netCDF4 file and saves all data into it."""
    dims = {
        "time": len(data_obs.radar.time),
        "height": len(data_obs.radar.height),
        "model_time": len(data_obs.model.time),
        "model_height": len(data_obs.model.mean_height),
    }

    file_type = "categorize"
    if "liquid_prob" in cloudnet_arrays:
        file_type += "-voodoo"

    with output.init_file(full_path, dims, cloudnet_arrays, uuid) as nc:
        uuid_out = nc.file_uuid
        nc.cloudnet_file_type = file_type
        output.copy_global(
            data_obs.radar.dataset,
            nc,
            ("year", "month", "day", "location"),
        )
        nc.title = f"Cloud categorization products from {data_obs.radar.location}"
        nc.source_file_uuids = output.get_source_uuids(data_obs)
        is_voodoo = "liquid_prob" in cloudnet_arrays
        extra_references = (
            ["https://doi.org/10.5194/amt-15-5343-2022"] if is_voodoo else None
        )
        nc.references = output.get_references(
            identifier=file_type,
            extra=extra_references,
        )
        if is_voodoo:
            import voodoonet.version

            nc.voodoonet_version = voodoonet.version.__version__
        output.add_source_instruments(nc, data_obs)
        output.merge_history(nc, file_type, data_obs)
    return uuid_out


def _classes_to_bits(data: QualityBits | CategoryBits) -> NDArray[np.int_]:
    shape = data.radar.shape if hasattr(data, "radar") else data.droplet.shape
    quality = np.zeros(shape, dtype=np.int64)
    for i, field in enumerate(dataclasses.fields(data)):
        quality |= (1 << i) * getattr(data, field.name)
    return quality


COMMENTS = {
    "category_bits": (
        "This variable contains information on the nature of the targets\n"
        "at each pixel, thereby facilitating the application of algorithms that work\n"
        "with only one type of target. The information is in the form of an array of\n"
        "bits, each of which states either whether a certain type of particle is\n"
        "present (e.g. aerosols), or the whether some of the target particles have\n"
        "a particular property. The definitions of each bit are given in the\n"
        "definition attribute. Bit 0 is the least significant."
    ),
    "quality_bits": (
        "This variable contains information on the quality of the data at each pixel.\n"
        "The information is in the form of an array of bits, and the definitions of\n"
        "each bit are given in the definition attribute. Bit 0 is the least\n"
        "significant."
    ),
    "lwp": (
        "This variable is the vertically integrated liquid water directly over\n"
        "the site. The temporal correlation of errors in liquid water path means that\n"
        "it is not really meaningful to distinguish bias from random error, so only\n"
        "an error variable is provided."
    ),
    "radar_liquid_atten": (
        "This variable was calculated from the liquid water path measured by\n"
        "microwave radiometer using lidar and radar returns to perform an\n"
        "approximate partitioning of the liquid water content with height. Bit 5\n"
        "of the quality_bits variable indicates where a correction for liquid water\n"
        "attenuation has been performed."
    ),
    "radar_gas_atten": (
        "This variable was calculated from the model temperature, pressure and\n"
        "humidity, but forcing pixels containing liquid cloud to saturation with\n"
        "respect to liquid water. It has been used to correct Z."
    ),
    "radar_rain_atten": (
        "This variable was calculated from the disdrometer rainfall rate."
    ),
    "radar_melting_atten": (
        "This variable was calculated from the disdrometer rainfall rate."
    ),
    "Tw": "This variable was derived from model temperature, pressure and humidity.",
    "Z_sensitivity": (
        "This variable is an estimate of the radar sensitivity, i.e. the minimum\n"
        "detectable radar reflectivity, as a function of height. It includes the\n"
        "effect of ground clutter and gas attenuation but not other attenuations."
    ),
    "Z_error": (
        "This variable is an estimate of the one-standard-deviation random error\n"
        "in radar reflectivity factor. It originates from the following independent\n"
        "sources of error:\n"
        "1) Precision in reflectivity estimate due to finite signal to noise and\n"
        "   finite number of pulses\n"
        "2) 10% uncertainty in gaseous attenuation correction (mainly due to error\n"
        "   in model humidity field)\n"
        "3) 20% uncertainty in rain attenuation correction (mainly due to error\n"
        "   in disdrometer rainfall rate)\n"
        "4) 10%-20% uncertainty in melting layer attenuation correction (mainly due\n"
        "   to error in disdrometer rainfall rate)\n"
        "5) Error in liquid water path (given by the variable lwp_error) and its\n"
        "   partitioning with height)."
    ),
    "Z": (
        "This variable has been corrected for attenuation by gaseous attenuation,\n"
        "and possibly liquid water, rain and melting layer (see quality_bits\n"
        "variable). Calibration convention: in the absence of attenuation, a cloud\n"
        "at 273 K containing one million 100-micron droplets per cubic metre will\n"
        "have\n a reflectivity of 0 dBZ at all frequencies."
    ),
    "bias": (
        "This variable is an estimate of the one-standard-deviation\n"
        "calibration error."
    ),
    "insect_prob": (
        "Ad-hoc estimation of the probability that the pixel contains insects."
    ),
    "liquid_prob": (
        "Probability derived from the radar data that the pixel contains liquid."
    ),
}

DEFINITIONS = {
    "category_bits": utils.bit_field_definition(
        {
            0: """Small liquid droplets are present.""",
            1: """Falling hydrometeors are present; if Bit 2 is set then these
                  are most likely ice particles, otherwise they are drizzle or
                  rain drops.""",
            2: """Wet-bulb temperature is less than 0 degrees C, implying the
                  phase of Bit-1 particles.""",
            3: """Melting ice particles are present.""",
            4: """Aerosol particles are present and visible to the lidar.""",
            5: """Insects are present and visible to the radar.""",
        }
    ),
    "quality_bits": utils.bit_field_definition(
        {
            0: """An echo is detected by the radar.""",
            1: """An echo is detected by the lidar.""",
            2: """The apparent echo detected by the radar is ground clutter or
                  some other non-atmospheric artifact.""",
            3: """The lidar echo is due to clear-air molecular scattering.""",
            4: """Liquid water cloud, rainfall or melting ice below this pixel
                  will have caused radar and lidar attenuation; if bit 5 is set
                  then a correction for the radar attenuation has been
                  performed; otherwise do not trust the absolute values of
                  reflectivity factor. No correction is performed for lidar
                  attenuation.""",
            5: """Radar reflectivity has been corrected for liquid-water
                  attenuation using the microwave radiometer measurements of
                  liquid water path and the lidar estimation of the location of
                  liquid water cloud; be aware that errors in reflectivity may
                  result.""",
            6: """Rain has caused radar attenuation; if bit 7 is set then a
                  correction for the radar attenuation has been performed;
                  otherwise do not trust the absolute values of reflectivity
                  factor. No correction is performed for lidar attenuation.""",
            7: """Radar reflectivity has been corrected for rain attenuation
                  using rainfall rate from a disdrometer; be aware that errors
                  in reflectivity may result.""",
            8: """Melting layer has caused radar attenuation; if bit 9 is set then a
                  correction for the radar attenuation has been performed;
                  otherwise do not trust the absolute values of reflectivity
                  factor. No correction is performed for lidar attenuation.""",
            9: """Radar reflectivity has been corrected for melting layer
                  attenuation; be aware that errors in reflectivity may result.""",
        }
    ),
}

CATEGORIZE_ATTRIBUTES = {
    # Radar variables
    "Z": MetaData(
        long_name="Radar reflectivity factor",
        units="dBZ",
        comment=COMMENTS["Z"],
        ancillary_variables="Z_error Z_bias Z_sensitivity",
    ),
    "Z_error": MetaData(
        long_name="Error in radar reflectivity factor",
        units="dB",
        comment=COMMENTS["Z_error"],
    ),
    "Z_bias": MetaData(
        long_name="Bias in radar reflectivity factor",
        units="dB",
        comment=COMMENTS["bias"],
    ),
    "Z_sensitivity": MetaData(
        long_name="Minimum detectable radar reflectivity",
        units="dBZ",
        comment=COMMENTS["Z_sensitivity"],
    ),
    "v_sigma": MetaData(
        long_name="Standard deviation of mean Doppler velocity",
        units="m s-1",
    ),
    # Lidar variables
    "beta": MetaData(
        long_name="Attenuated backscatter coefficient",
        units="sr-1 m-1",
        comment="SNR-screened attenuated backscatter coefficient.",
        ancillary_variables="beta_error beta_bias",
    ),
    "beta_error": MetaData(
        long_name="Error in attenuated backscatter coefficient",
        units="dB",
    ),
    "beta_bias": MetaData(
        long_name="Bias in attenuated backscatter coefficient",
        units="dB",
    ),
    "lidar_wavelength": MetaData(long_name="Laser wavelength", units="nm"),
    # MWR variables
    "lwp_error": MetaData(
        long_name="Error in liquid water path",
        units="kg m-2",
    ),
    "lwp": MetaData(ancillary_variables="lwp_error"),
    # Model variables
    "Tw": MetaData(long_name="Wet-bulb temperature", units="K", comment=COMMENTS["Tw"]),
    "model_time": MetaData(long_name="Model time UTC", calendar="standard"),
    "model_height": MetaData(
        long_name="Height of model variables above mean sea level",
        units="m",
        axis="Z",
    ),
    "vwind": MetaData(
        long_name="Meridional wind",
        units="m s-1",
    ),
    "uwind": MetaData(
        long_name="Zonal wind",
        units="m s-1",
    ),
    "q": MetaData(long_name="Specific humidity", units="1"),
    # MISC
    "category_bits": MetaData(
        long_name="Target categorization bits",
        comment=COMMENTS["category_bits"],
        definition=DEFINITIONS["category_bits"],
        units="1",
    ),
    "quality_bits": MetaData(
        long_name="Data quality bits",
        comment=COMMENTS["quality_bits"],
        definition=DEFINITIONS["quality_bits"],
        units="1",
    ),
    "radar_liquid_atten": MetaData(
        long_name="Two-way radar attenuation due to liquid water",
        units="dB",
        comment=COMMENTS["radar_liquid_atten"],
        references="ITU-R P.840-9",
    ),
    "radar_rain_atten": MetaData(
        long_name="Two-way radar attenuation due to rain",
        units="dB",
        references="Crane, R. (1980)",
        comment=COMMENTS["radar_rain_atten"],
    ),
    "radar_melting_atten": MetaData(
        long_name="Two-way radar attenuation due to melting ice",
        units="dB",
        references="Li, H., & Moisseev, D. (2019)",
        comment=COMMENTS["radar_melting_atten"],
    ),
    "radar_gas_atten": MetaData(
        long_name="Two-way radar attenuation due to atmospheric gases",
        units="dB",
        comment=COMMENTS["radar_gas_atten"],
        references="ITU-R P.676-13",
    ),
    "insect_prob": MetaData(
        long_name="Insect probability",
        units="1",
        comment=COMMENTS["insect_prob"],
    ),
    "liquid_prob": MetaData(
        long_name="Liquid probability",
        units="1",
        comment=COMMENTS["liquid_prob"],
    ),
    "rain_detected": MetaData(
        long_name="Rain detected",
        units="1",
        comment="1 = rain detected, 0 = no rain detected",
    ),
}
