import numpy as np
import numpy.ma as ma
from cloudnetpy.categorize import DataSource
from cloudnetpy import utils
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy import atmos
from cloudnetpy import plotting
import matplotlib as mpl
import matplotlib.pyplot as plt


G_TO_KG = 0.001


class LwcSource(DataSource):
    """Data container for liquid water content calculations."""
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.atmosphere = self._get_atmosphere()
        self.lwp = self.getvar('lwp')
        self.lwp_error = self.getvar('lwp_error')
        self.dheight = utils.mdiff(self.getvar('height'))
        self.is_rain = self.getvar('is_rain')

    def _get_atmosphere(self):
        return (self._interpolate_model_field('temperature'),
                self._interpolate_model_field('pressure'))

    def _interpolate_model_field(self, variable_name):
        """Interpolates 2D model field into Cloudnet grid."""
        return utils.interpolate_2d(self.getvar('model_time'),
                                    self.getvar('model_height'),
                                    self.getvar(variable_name),
                                    self.time, self.getvar('height'))


class Lwc:
    """Class handling the liquid water content calculations."""
    def __init__(self, categorize_file):
        self.lwc_input = LwcSource(categorize_file)
        self.echo = self._get_echo()
        self.is_liquid = self._get_liquid()
        self.lwc_adiabatic = self._init_lwc_adiabatic()  # g/m3
        self.lwc = self._init_lwc()
        self.status = self._init_status()

    def _get_echo(self):
        quality_bits = p_tools.read_quality_bits(self.lwc_input)
        return {'radar': quality_bits['radar'], 'lidar': quality_bits['lidar']}

    def _get_liquid(self):
        category_bits = p_tools.read_category_bits(self.lwc_input)
        return category_bits['droplet']

    def _init_lwc_adiabatic(self):
        """Returns theoretical adiabatic lwc in liquid clouds (g/m3)."""
        lwc_dz = atmos.fill_clouds_with_lwc_dz(self.lwc_input.atmosphere,
                                               self.is_liquid)
        return atmos.calc_adiabatic_lwc(lwc_dz, self.lwc_input.dheight)

    def _init_lwc(self):
        """Initialises liquid water content (g/m3).

        Calculates LWC for ALL profiles (rain, lwp > theoretical, etc.),
        """
        lwc_scaled = atmos.distribute_lwp_to_liquid_clouds(self.lwc_adiabatic,
                                                           self.lwc_input.lwp)
        return lwc_scaled / self.lwc_input.dheight

    def _init_status(self):
        status = ma.zeros(self.lwc.shape, dtype=int)
        status[self.lwc_adiabatic > 0] = 1
        return status

    def adjust_clouds_to_match_measured_lwp(self):
        no_rain = ~self.lwc_input.is_rain.astype(bool)
        lwp_difference = self._find_lwp_difference()
        adjustable_clouds = find_adjustable_clouds(self.is_liquid, self.echo)
        dubious_profiles = (lwp_difference < 0) & no_rain
        adjustable_clouds[~dubious_profiles, :] = 0
        self.lwc_adiabatic = adjust_cloud_tops(adjustable_clouds,
                                               self.lwc_adiabatic,
                                               lwp_difference,
                                               self.lwc_input.dheight)

    def _find_lwp_difference(self):
        """Returns difference of theoretical LWP and measured LWP.

        In theory, this difference should be always positive. Negative values
        indicate missing (or too narrow) liquid clouds.
        """
        lwc_sum = ma.sum(self.lwc_adiabatic, axis=1) * self.lwc_input.dheight
        return lwc_sum - self.lwc_input.lwp


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


def generate_lwc(categorize_file):
    """High level API to generate Cloudnet liquid water content file."""
    lwc = Lwc(categorize_file)
    lwc.adjust_clouds_to_match_measured_lwp()
