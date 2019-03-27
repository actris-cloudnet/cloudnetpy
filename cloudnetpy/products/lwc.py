"""Module for creating Cloudnet liquid water content file
using scaled-adiabatic method.
"""
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils, atmos, output
from cloudnetpy.categorize import DataSource
from cloudnetpy.products import product_tools as p_tools


G_TO_KG = 0.001


class LwcSource(DataSource):
    """Data container for liquid water content calculations. It reads
    input data from a categorize file and provides data structures and
    methods for holding the results.
    """
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
    """Class handling the actual LWC calculations."""
    def __init__(self, lwc_input):
        self.lwc_input = lwc_input
        self.dheight = self.lwc_input.dheight
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
        return atmos.calc_adiabatic_lwc(lwc_dz, self.dheight)

    def _adiabatic_lwc_to_lwc(self):
        """Initialises liquid water content (g/m3).

        Calculates LWC for ALL profiles (rain, lwp > theoretical, etc.),
        """
        lwc_scaled = atmos.distribute_lwp_to_liquid_clouds(self.lwc_adiabatic,
                                                           self.lwc_input.lwp)
        return lwc_scaled / self.dheight

    def _init_status(self):
        status = ma.zeros(self.is_liquid.shape, dtype=int)
        status[self.is_liquid] = 1
        return status

    def adjust_clouds_to_match_lwp(self):
        """Adjust clouds (where possible) so that theoretical and measured LWP agree."""
        adjustable_clouds = self._find_adjustable_clouds()
        self._adjust_cloud_tops(adjustable_clouds)
        self.lwc = self._adiabatic_lwc_to_lwc()

    def _find_lwp_difference(self):
        """Returns difference of theoretical LWP and measured LWP (g/m2).

        In theory, this difference should be always positive. Negative values
        indicate missing (or too narrow) liquid clouds.
        """
        lwc_sum = ma.sum(self.lwc_adiabatic, axis=1) * self.dheight
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

        def _remove_good_profiles():
            no_rain = ~self.lwc_input.is_rain.astype(bool)
            lwp_difference = self._find_lwp_difference()
            dubious_profiles = (lwp_difference < 0) & no_rain
            top_clouds[~dubious_profiles, :] = 0

        top_clouds = find_topmost_clouds(self.is_liquid)
        detection_type = _find_echo_combinations_in_liquid()
        detection_type[~top_clouds] = 0
        lidar_only_clouds = _find_lidar_only_clouds(detection_type)
        top_clouds[~lidar_only_clouds, :] = 0
        _remove_good_profiles()
        return top_clouds

    def _adjust_cloud_tops(self, adjustable_clouds):
        """Adjusts cloud top index so that measured lwc corresponds to
        theoretical value.
        """
        def _has_converged(time_ind):
            lwc_sum = ma.sum(self.lwc_adiabatic[time_ind, :])
            if lwc_sum * self.dheight > self.lwc_input.lwp[time_ind]:
                return True
            return False

        def _adjust_lwc(time_ind, base_ind):
            lwc_base = self.lwc_adiabatic[time_ind, base_ind]
            distance_from_base = 1
            while True:
                top_ind = base_ind + distance_from_base
                lwc_top = lwc_base * (distance_from_base + 1)
                self.lwc_adiabatic[time_ind, top_ind] = lwc_top
                if not self.status[time_ind, top_ind]:
                    self.status[time_ind, top_ind] = 3
                if _has_converged(time_ind):
                    break
                distance_from_base += 1

        def _update_status(time_ind):
            alt_indices = np.where(self.is_liquid[time_ind, :])[0]
            self.status[time_ind, alt_indices] = 2

        for time_index in np.unique(np.where(adjustable_clouds)[0]):
            base_index = np.where(adjustable_clouds[time_index, :])[0][0]
            _update_status(time_index)
            _adjust_lwc(time_index, base_index)

    def screen_rain(self):
        """Masks profiles with rain."""
        is_rain = self.lwc_input.is_rain.astype(bool)
        self.lwc[is_rain, :] = ma.masked
        self.status[is_rain, :] = 6


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


def generate_lwc(categorize_file, output_file):
    """High level API to generate Cloudnet liquid water content file."""
    lwc_data = LwcSource(categorize_file)
    lwc_obj = Lwc(lwc_data)
    lwc_obj.adjust_clouds_to_match_lwp()
    lwc_obj.screen_rain()
    _append_data(lwc_data, lwc_obj)
    output.update_attributes(lwc_data.data)
    _save_data_and_meta(lwc_data, output_file)


def _append_data(lwc_data, lwc_obj):
    lwc_data.append_data(lwc_obj.lwc, 'lwc', units='g m-3')
    lwc_data.append_data(lwc_obj.status, 'lwc_retrieval_status')
    lwc_data.append_data(lwc_data.lwp, 'lwp')
    lwc_data.append_data(lwc_data.lwp_error, 'lwp_error')


def _save_data_and_meta(lwc_data, output_file):
    """
    Saves wanted information to NetCDF file.
    """
    dims = {'time': len(lwc_data.time),
            'height': len(lwc_data.variables['height'])}
    rootgrp = output.init_file(output_file, dims, lwc_data.data, zlib=True)
    vars_from_source = ('altitude', 'latitude', 'longitude', 'time', 'height')
    output.copy_variables(lwc_data.dataset, rootgrp, vars_from_source)
    rootgrp.title = f"Liquid water content file from {lwc_data.dataset.location}"
    rootgrp.source = f"Categorize file: {p_tools.get_source(lwc_data)}"
    output.copy_global(lwc_data.dataset, rootgrp, ('location', 'day',
                                                   'month', 'year'))
    output.merge_history(rootgrp, 'liquid water content', lwc_data)
    rootgrp.close()
