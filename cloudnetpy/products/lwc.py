import numpy as np
import numpy.ma as ma
from cloudnetpy.categorize import DataSource
from cloudnetpy import utils
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy import plotting
from cloudnetpy import atmos
import matplotlib as mpl
import matplotlib.pyplot as plt


G_TO_KG = 0.001


class LwcSource(DataSource):
    """Data container for liquid water content calculations."""
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.temperature = self._interpolate_model_field('temperature')
        self.pressure = self._interpolate_model_field('pressure')
        self.lwp = self.getvar('lwp')
        self.lwp_error = self.getvar('lwp_error')
        self.dheight = utils.mdiff(self.getvar('height'))
        self.rain_in_profile = self.getvar('is_rain')

    def _interpolate_model_field(self, variable_name):
        """Interpolates 2D model field into Cloudnet grid."""
        return utils.interpolate_2d(self.getvar('model_time'),
                                    self.getvar('model_height'),
                                    self.getvar(variable_name),
                                    self.time, self.getvar('height'))


class Liquid:
    """Data container for liquid droplets."""
    def __init__(self, categorize_object):
        self.category_bits = p_tools.read_category_bits(categorize_object)
        self.quality_bits = p_tools.read_quality_bits(categorize_object)
        self.is_liquid = self.category_bits['droplet']


class Lwc:
    """Class handling the liquid water content calculations."""
    def __init__(self, lwc_input, liquid):
        self.lwc_input = lwc_input
        self.liquid = liquid
        self.lwp = lwc_input.lwp
        self.lwc_adiabatic = None
        self.lwc_scaled = None
        self._init_lwc()

    def _init_lwc(self):
        atmosphere = (self.lwc_input.temperature, self.lwc_input.pressure)
        is_liquid = self.liquid.is_liquid
        dheight = self.lwc_input.dheight
        lwc_change_rate = atmos.fill_clouds_with_lwc_dz(atmosphere, is_liquid)
        self.lwc_adiabatic = atmos.calc_adiabatic_lwc(lwc_change_rate, dheight)
        self.lwc_scaled = atmos.scale_lwc(self.lwc_adiabatic, self.lwp) * G_TO_KG


def init_status(categorize_object, lwc_object):

    category_bits = p_tools.read_category_bits(categorize_object)
    quality_bits = p_tools.read_quality_bits(categorize_object)
    no_rain = ~categorize_object.rain_in_profile.astype(bool)
    dheight = categorize_object.dheight
    lwp = lwc_object.lwp / dheight
    lwc_adiabatic = lwc_object.lwc_adiabatic
    status = ma.zeros(lwc_adiabatic.shape, dtype=int)
    lwc_sum = ma.sum(lwc_adiabatic, axis=1)
    lwc = atmos.scale_lwc(lwc_adiabatic, lwp)
    lwc_difference = lwc_sum - lwp
    good_profiles = (lwc_difference > 0) & no_rain

    # These are valid lwc-values and they seem to correct
    lwc[~good_profiles, :] = ma.masked
    status[lwc > 0] = 1

    # now suspicious profiles that we maybe can adjust
    is_liquid = category_bits['droplet']
    echo = {'radar': quality_bits['radar'], 'lidar': quality_bits['lidar']}
    adjustable_clouds = find_adjustable_clouds(is_liquid, echo)
    dubious_profiles = (lwc_difference < 0) & no_rain
    adjustable_clouds[~dubious_profiles, :] = 0
    lwc_adiabatic = adjust_cloud_tops(adjustable_clouds, lwc_adiabatic, lwc_difference, dheight)
    plotting.plot_2d(lwc_adiabatic)


def adjust_cloud_tops(adjustable_clouds, lwc_adiabatic, lwc_difference, dheight):
    """Adjusts cloud top index so that measured lwc corresponds to
    theoretical value.
    """
    for time_ind in np.unique(np.where(adjustable_clouds)[0]):
        cloud_inds = np.where(adjustable_clouds[time_ind, :])[0]
        base_ind = cloud_inds[0]
        lwc_dz = lwc_adiabatic[time_ind, base_ind] * dheight
        difference = np.abs(lwc_difference[time_ind])
        n_steps_needed = np.sqrt(2*(1/lwc_dz)*difference)
        adjusted_lwc = lwc_dz * np.arange(n_steps_needed)
        lwc_adiabatic[time_ind, base_ind:base_ind+len(adjusted_lwc)] = adjusted_lwc
    return lwc_adiabatic


def find_adjustable_clouds(is_liquid, echo):
    top_clouds = find_topmost_clouds(is_liquid)
    detection_type = find_echo_combinations_in_liquid(is_liquid, echo)
    detection_type[~top_clouds] = 0
    lidar_only = find_lidar_only_top_clouds(detection_type)
    top_clouds[~lidar_only, :] = 0
    return top_clouds


def find_lidar_only_top_clouds(detection_type):
    """Finds top clouds that contain only lidar-detected pixels.

    Args:
        detection_type (ndarray): Array of integers where 1=lidar, 2=radar,
            3=both.

    Returns:
        ndarray: Boolean array containing top-clouds that are detected only
            by lidar.

    """
    sum_of_cloud_pixels = ma.sum(detection_type > 0, axis=1)
    sum_of_detection_type = ma.sum(detection_type, axis=1)
    return sum_of_cloud_pixels / sum_of_detection_type == 1


def find_echo_combinations_in_liquid(is_liquid, echo):
    """Classifies liquid clouds by detection type: 1=lidar, 2=radar, 3=both.

    Args:
        is_liquid (ndarray): Boolean array denoting classified liquid layers.
        echo (dict): Dict containing 'lidar' and 'radar' which are boolean
            arrays of denoting detection.

    Returns:
        ndarray: Array of integers where 1=lidar, 2=radar, 3=both.

    """
    lidar_detected_liquid = (is_liquid & echo['lidar']).astype(int)
    radar_detected_liquid = (is_liquid & echo['radar']).astype(int) * 2
    return lidar_detected_liquid + radar_detected_liquid


def find_topmost_clouds(is_cloud):
    """From 2d binary cloud field, return the uppermost cloud layer only.

    Args:
        is_cloud (ndarray): Boolean array denoting presence of clouds.

    Returns:
        ndarray: Copy of input array containing only the uppermost cloud
             layer in each profile.

    """
    top_clouds = np.copy(is_cloud)
    cloud_edges = top_clouds[:, :-1][:, ::-1] < top_clouds[:, 1:][:, ::-1]
    topmost_bases = is_cloud.shape[1] - 1 - np.argmax(cloud_edges, axis=1)
    for n, base in enumerate(topmost_bases):
        top_clouds[n, :base] = 0
    return top_clouds


def generate_lwc(categorize_file):
    """High level API to generate Cloudnet liquid water content file."""
    lwc_input = LwcSource(categorize_file)
    liquid = Liquid(lwc_input)
    lwc = Lwc(lwc_input, liquid)

    init_status(lwc_input, lwc)

    #liquid = Liquid(lwc_input)
    #lwc = Lwc(lwc_input, liquid)
    #import netCDF4
    #refe = netCDF4.Dataset('/home/tukiains/Documents/PYTHON/cloudnetpy/test_data/20181204_mace-head_lwc-scaled-adiabatic.nc').variables['lwc'][:]
    #plotting.plot_2d(np.log10(refe), cmap='jet', clim=(-5, -2))
