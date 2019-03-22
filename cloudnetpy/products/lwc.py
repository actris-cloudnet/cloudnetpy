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
        self.lwc_scaled = self._get_lwc()

    def _get_lwc(self):
        atmosphere = (self.lwc_input.temperature, self.lwc_input.pressure)
        is_liquid = self.liquid.is_liquid
        dheight = self.lwc_input.dheight
        lwc_change_rate = atmos.fill_clouds_with_lwc_dz(atmosphere, is_liquid)
        lwc = atmos.calc_adiabatic_lwc(lwc_change_rate, dheight)
        self.lwc_adiabatic = lwc
        lwc_scaled = atmos.scale_lwc(lwc, self.lwp) * G_TO_KG
        return lwc_scaled


def init_status(categorize_object, lwc_object):

    category_bits = p_tools.read_category_bits(categorize_object)
    quality_bits = p_tools.read_quality_bits(categorize_object)

    no_rain = ~categorize_object.rain_in_profile.astype(bool)
    dheight = categorize_object.dheight
    lwp = lwc_object.lwp / dheight
    lwc_adiabatic = lwc_object.lwc_adiabatic
    lwc = ma.zeros(lwc_adiabatic.shape)
    status = ma.zeros(lwc_adiabatic.shape, dtype=int)
    lwc_sum = ma.sum(lwc_adiabatic, axis=1)
    lwc = atmos.scale_lwc(lwc_adiabatic, lwp)

    good_profiles = (lwc_sum > lwp) & no_rain

    # These are valid lwc-values and they seem to correct
    lwc[~good_profiles, :] = ma.masked
    status[lwc > 0] = 1

    # now suspicious profiles that we maybe can adjust
    is_liquid = category_bits['droplet']
    echo = {'radar': quality_bits['radar'], 'lidar': quality_bits['lidar']}

    dubious_profiles = (lwc_sum < lwp) & no_rain

    # adjust status-5 clouds in bad profiles. They are at top.
    adjustable_clouds = find_status5_clouds(is_liquid, echo)

    ind = dubious_profiles & (np.any(adjustable_clouds, axis=1))

    print(np.sum(ind))


def find_status5_clouds(is_liquid, echo):
    top_clouds = find_topmost_clouds(is_liquid)
    detection_type = find_echo_combinations_in_liquid(is_liquid, echo)
    detection_type[~top_clouds] = 0
    lidar_only = find_lidar_only_top_clouds(detection_type)
    top_clouds[~lidar_only, :] = 0
    return top_clouds


def find_lidar_only_top_clouds(detection_type):
    sum_of_cloud_pixels = ma.sum(detection_type > 0, axis=1)
    sum_of_detection_type = ma.sum(detection_type, axis=1)
    return sum_of_cloud_pixels / sum_of_detection_type == 1


def find_echo_combinations_in_liquid(is_liquid, echo):
    lidar_detected_cloud = (is_liquid & echo['lidar']).astype(int)
    radar_detected_cloud = (is_liquid & echo['radar']).astype(int) * 2
    return lidar_detected_cloud + radar_detected_cloud


def find_topmost_clouds(is_cloud):
    """From 2d binary cloud field, return the topmost clouds only."""
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
