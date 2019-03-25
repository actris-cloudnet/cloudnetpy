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
        self.lwc_adiabatic = self._init_lwc_adiabatic()
        self.lwc = self._adiabatic_lwc_to_lwc()
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

    def _adiabatic_lwc_to_lwc(self):
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
        adjustable_clouds = self._find_adjustable_clouds()
        dubious_profiles = (lwp_difference < 0) & no_rain
        adjustable_clouds[~dubious_profiles, :] = 0
        self._adjust_cloud_tops(adjustable_clouds, lwp_difference)
        self.lwc = self._adiabatic_lwc_to_lwc()

    def _find_lwp_difference(self):
        """Returns difference of theoretical LWP and measured LWP.

        In theory, this difference should be always positive. Negative values
        indicate missing (or too narrow) liquid clouds.
        """
        lwc_sum = ma.sum(self.lwc_adiabatic, axis=1) * self.lwc_input.dheight
        return lwc_sum - self.lwc_input.lwp

    def _find_adjustable_clouds(self):

        def _find_echo_combinations_in_liquid():
            """Classifies liquid clouds by detection type: 1=lidar, 2=radar, 3=both."""
            lidar_detected = (self.is_liquid & self.echo['lidar']).astype(int)
            radar_detected = (self.is_liquid & self.echo['radar']).astype(int) * 2
            return lidar_detected + radar_detected

        def _find_lidar_only_clouds(detection):
            """Finds top clouds that contain only lidar-detected pixels.

            Args:
                detection_type (ndarray): Array of integers where 1=lidar, 2=radar,
                3=both.

            Returns:
                ndarray: Boolean array containing top-clouds that are detected only
                by lidar.

            """
            sum_of_cloud_pixels = ma.sum(detection > 0, axis=1)
            sum_of_detection_type = ma.sum(detection, axis=1)
            return sum_of_cloud_pixels / sum_of_detection_type == 1

        top_clouds = find_topmost_clouds(self.is_liquid)
        detection_type = _find_echo_combinations_in_liquid()
        detection_type[~top_clouds] = 0
        lidar_only_clouds = _find_lidar_only_clouds(detection_type)
        top_clouds[~lidar_only_clouds, :] = 0
        return top_clouds

    def _adjust_cloud_tops(self, adjustable_clouds, lwc_difference):
        """Adjusts cloud top index so that measured lwc corresponds to
        theoretical value.
        """
        def _calc_steps_to_cover_area(derivative, area):
            n_steps = np.sqrt(2 * (1 / derivative) * area)
            return np.floor(n_steps).astype(int)

        def _adjust_lwc(difference, derivative):
            abs_difference = np.abs(difference)
            n_steps_needed = _calc_steps_to_cover_area(derivative, abs_difference)
            return derivative * np.arange(n_steps_needed)

        for time_ind in np.unique(np.where(adjustable_clouds)[0]):
            base_ind = np.where(adjustable_clouds[time_ind, :])[0][0]
            lwc_dz = self.lwc_adiabatic[time_ind, base_ind] * self.lwc_input.dheight
            adjusted_lwc = _adjust_lwc(lwc_difference[time_ind], lwc_dz)
            self.lwc_adiabatic[time_ind, base_ind:base_ind+len(adjusted_lwc)] = adjusted_lwc


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
    lwc = Lwc(categorize_file)
    lwc.adjust_clouds_to_match_measured_lwp()
