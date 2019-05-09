"""Module for creating Cloudnet liquid water content file
using scaled-adiabatic method.
"""
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils, atmos, output
from cloudnetpy.categorize import DataSource
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import CategorizeBits

G_TO_KG = 0.001


class LwcSource(DataSource):
    """Data container for liquid water content calculations. It reads
    input data from a categorize file and provides data structures and
    methods for holding the results.
    """
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.lwp = self.getvar('lwp')
        self.lwp_error = self.getvar('lwp_error')
        self.is_rain = self.getvar('is_rain')
        self.dheight = utils.mdiff(self.getvar('height'))
        self.atmosphere = self._get_atmosphere(categorize_file)
        self.categorize_bits = CategorizeBits(categorize_file)

    @staticmethod
    def _get_atmosphere(categorize_file):
        fields = ['temperature', 'pressure']
        return p_tools.interpolate_model(categorize_file, fields)


class Lwc:
    """Class handling the actual LWC calculations."""
    def __init__(self, lwc_source):
        self.lwc_source = lwc_source
        self.dheight = self.lwc_source.dheight
        self.echo = self._get_echo()
        self.is_liquid = self._get_liquid()
        self.lwc_adiabatic = self._init_lwc_adiabatic()
        self.lwc = self._adiabatic_lwc_to_lwc()
        self.status = self._init_status()
        self.lwc_error = self._calc_lwc_error()

    def _calc_lwc_error(self):
        """Estimates error in liquid water content."""
        lwc_error = np.zeros_like(self.lwc)
        lwp_relative_error = self.lwc_source.lwp_error / self.lwc_source.lwp
        #lwp_relative_error[(lwp_relative_error < 0) | (lwp_relative_error > 10)] = 10  # not sure about this
        ind = np.where(self.lwc)
        lwc_gradient = np.gradient(self.lwc[ind])
        lwc_error[ind] = lwc_gradient ** 2
        lwc_error = lwc_error + utils.transpose(lwp_relative_error) ** 2
        lwc_error[self.lwc == 0] = ma.masked
        return ma.sqrt(lwc_error)

    def _get_echo(self):
        quality_bits = self.lwc_source.categorize_bits.quality_bits
        return {'radar': quality_bits['radar'], 'lidar': quality_bits['lidar']}

    def _get_liquid(self):
        category_bits = self.lwc_source.categorize_bits.category_bits
        return category_bits['droplet']

    def _init_lwc_adiabatic(self):
        """Returns theoretical adiabatic lwc in liquid clouds (g/m3)."""
        lwc_dz = atmos.fill_clouds_with_lwc_dz(self.lwc_source.atmosphere,
                                               self.is_liquid)
        return atmos.calc_adiabatic_lwc(lwc_dz, self.dheight)

    def _adiabatic_lwc_to_lwc(self):
        """Initialises liquid water content (g/m3).

        Calculates LWC for ALL profiles (rain, lwp > theoretical, etc.),
        """
        lwc_scaled = atmos.distribute_lwp_to_liquid_clouds(self.lwc_adiabatic,
                                                           self.lwc_source.lwp)
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
        return lwc_sum - self.lwc_source.lwp

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
            no_rain = ~self.lwc_source.is_rain.astype(bool)
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
            if lwc_sum * self.dheight > self.lwc_source.lwp[time_ind]:
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
        is_rain = self.lwc_source.is_rain.astype(bool)
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
    """High level API to generate Cloudnet liquid water content product.

    Args:
        categorize_file (str): Categorize file name.
        output_file (str): Output file name.

    Examples:
        >>> from cloudnetpy.products.lwc import generate_lwc
        >>> generate_lwc('categorize.nc', 'lwc.nc')

    """
    lwc_source = LwcSource(categorize_file)
    lwc_obj = Lwc(lwc_source)
    lwc_obj.adjust_clouds_to_match_lwp()
    lwc_obj.screen_rain()
    _append_data(lwc_source, lwc_obj)
    output.update_attributes(lwc_source.data, LWC_ATTRIBUTES)
    _save_data_and_meta(lwc_source, output_file)


def _append_data(lwc_data, lwc_obj):
    lwc_data.append_data(lwc_obj.lwc * G_TO_KG, 'lwc', units='kg m-3')
    lwc_data.append_data(lwc_obj.lwc_error * G_TO_KG, 'lwc_error', units='kg m-3')
    lwc_data.append_data(lwc_obj.status, 'lwc_retrieval_status')


def _save_data_and_meta(lwc_data, output_file):
    """
    Saves wanted information to NetCDF file.
    """
    dims = {'time': len(lwc_data.time),
            'height': len(lwc_data.variables['height'])}
    rootgrp = output.init_file(output_file, dims, lwc_data.data, zlib=True)
    vars_from_source = ('altitude', 'latitude', 'longitude', 'time', 'height', 'lwp', 'lwp_error')
    output.copy_variables(lwc_data.dataset, rootgrp, vars_from_source)
    rootgrp.title = f"Liquid water content file from {lwc_data.dataset.location}"
    rootgrp.source = f"Categorize file: {p_tools.get_source(lwc_data)}"
    output.copy_global(lwc_data.dataset, rootgrp, ('location', 'day',
                                                   'month', 'year'))
    output.merge_history(rootgrp, 'liquid water content', lwc_data)
    rootgrp.close()


COMMENTS = {
    'lwc':
        ('This variable was calculated for the profiles where the categorization\n'
         'data has diagnosed that liquid water is present and liquid water path is\n'
         'available from a coincident microwave radiometer. The model temperature\n'
         'and pressure were used to estimate the theoretical adiabatic liquid water\n'
         'content gradient for each cloud base and the adiabatic liquid water\n'
         'content is then scaled that its integral matches the radiometer\n'
         'measurement so that the liquid water content now follows a quasi-adiabatic\n'
         'profile. If the liquid layer is detected by the lidar only, there is the\n'
         'potential for cloud top height to be underestimated and so if the\n'
         'adiabatic integrated liquid water content is less than that measured by\n'
         'the microwave radiometer, the cloud top is extended until the adiabatic\n'
         'integrated liquid water content agrees with the value measured by the\n'
         'microwave radiometer. Missing values indicate that either\n'
         '1) a liquid water layer was diagnosed but no microwave radiometer data was\n'
         '   available,\n'
         '2) a liquid water layer was diagnosed but the microwave radiometer data\n'
         '   was unreliable; this may be because a melting layer was present in the\n'
         '   profile, or because the retrieved lwp was unphysical (values of zero\n'
         '   are not uncommon for thin supercooled liquid layers)\n'
         '3) that rain is present in the profile and therefore, the vertical extent\n'
         '   of liquid layers is difficult to ascertain.'),

    'lwc_error':
        ('This variable is an estimate of the random error in liquid water content\n'
         'due to the uncertainty in the microwave radiometer liquid water path\n'
         'retrieval and the uncertainty in cloud base and/or cloud top height.\n'
         'This is associated with the resolution of the grid used, 20 m,\n'
         'which can affect both cloud base and cloud top. If the liquid layer is\n'
         'detected by the lidar only, there is the potential for cloud top height\n'
         'to be underestimated. Similarly, there is the possibility that the lidar\n'
         'may not detect the second cloud base when multiple layers are present and\n'
         'the cloud base will be overestimated. It is assumed that the error\n'
         'contribution arising from using the model temperature and pressure at\n'
         'cloud base is negligible.'),

    'lwc_retrieval_status':
        ('This variable describes whether a retrieval was performed for each pixel,\n'
         'and its associated quality, in the form of 6 different classes. The classes\n'
         'are defined in the definition and long_definition attributes.\n'
         'The most reliable retrieval is that when both radar and lidar detect the\n'
         'liquid layer, and microwave radiometer data is present, indicated by the\n'
         'value 1. The next most reliable is when microwave radiometer data is used\n'
         'to adjust the cloud depth when the radar does not detect the liquid layer,\n'
         'indicated by the value 2, with a value of 3 indicating the cloud pixels\n'
         'that have been added at cloud top to avoid the profile becoming\n'
         'superadiabatic. A value of 4 indicates that microwave radiometer data\n'
         'were not available or not reliable (melting level present or unphysical\n'
         'values) but the liquid layers were well defined.  If cloud top was not\n'
         'well defined then this is indicated by a value of 5. The full retrieval of\n'
         'liquid water content, which requires reliable liquid water path from the\n'
         'microwave radiometer, was only performed for classes 1-3. No attempt is\n'
         'made to retrieve liquid water content when rain is present; this is\n'
         'indicated by the value 6.'),
}

DEFINITIONS = {
    'lwc_retrieval_status':
        ('\n'
         'Value 0: No liquid water detected\n'
         'Value 1: Reliable retrieval\n'
         'Value 2: Adiabatic retrieval where cloud top has been adjusted to match\n'
         '         liquid water path from microwave radiometer because layer is not\n'
         '         detected by radar.\n'
         'Value 3: Adiabatic retrieval: new cloud pixels where cloud top has been\n'
         '         adjusted to match liquid water path from microwave radiometer\n'
         '         because layer is not detected by radar.\n'
         'Value 4: No retrieval: either no liquid water path is available or liquid\n'
         '         water path is uncertain.\n'
         'Value 5: No retrieval: liquid water layer detected only by the lidar and\n'
         '         liquid water path is unavailable or uncertain:\n'
         '         cloud top may be higher than diagnosed cloud top since lidar\n'
         '         signal has been attenuated.\n'
         'Value 6: Rain present: cloud extent is difficult to ascertain and liquid\n'
         '         water path also uncertain.'),
}

LWC_ATTRIBUTES = {
    'lwc': MetaData(
        long_name='Liquid water content',
        comment=COMMENTS['lwc'],
        ancillary_variables='lwc_error'
    ),
    'lwc_error': MetaData(
        long_name='Random error in liquid water content, one standard deviation',
        comment=COMMENTS['lwc_error'],
    ),
    'lwc_retrieval_status': MetaData(
        long_name='Liquid water content retrieval status',
        comment=COMMENTS['lwc_retrieval_status'],
        definition=DEFINITIONS['lwc_retrieval_status']
    ),
}
