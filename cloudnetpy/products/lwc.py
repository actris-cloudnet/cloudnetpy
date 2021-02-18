"""Module for creating Cloudnet liquid water content file
using scaled-adiabatic method.
"""
from typing import Union
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils, output
from cloudnetpy.categorize import atmos, DataSource
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy.metadata import MetaData
from cloudnetpy.products.product_tools import CategorizeBits

G_TO_KG = 0.001


def generate_lwc(categorize_file: str,
                 output_file: str,
                 keep_uuid: bool = False,
                 uuid: Union[str, None] = None) -> str:
    """Generates Cloudnet liquid water content product.

    This function calculates cloud liquid water content using the so-called
    adiabatic-scaled method. In this method, liquid water content measured by
    microwave radiometer is used to constrain the theoretical liquid water
    content of observed liquid clouds. The results are written in a netCDF file.

    Args:
        categorize_file (str): Categorize file name.
        output_file (str): Output file name.
        keep_uuid (bool, optional): If True, keeps the UUID of the old file,
            if that exists. Default is False when new UUID is generated.
        uuid (str, optional): Set specific UUID for the file.

    Returns:
        str: UUID of the generated file.

    Examples:
        >>> from cloudnetpy.products import generate_lwc
        >>> generate_lwc('categorize.nc', 'lwc.nc')

    References:
        Illingworth, A.J., R.J. Hogan, E. O'Connor, D. Bouniol, M.E. Brooks,
        J. Delanoé, D.P. Donovan, J.D. Eastment, N. Gaussiat, J.W. Goddard,
        M. Haeffelin, H.K. Baltink, O.A. Krasnov, J. Pelon, J. Piriou, A. Protat,
        H.W. Russchenberg, A. Seifert, A.M. Tompkins, G. van Zadelhoff, F. Vinit,
        U. Willén, D.R. Wilson, and C.L. Wrench, 2007: Cloudnet.
        Bull. Amer. Meteor. Soc., 88, 883–898, https://doi.org/10.1175/BAMS-88-6-883

    """
    lwc_source = LwcSource(categorize_file)
    lwc_obj = Lwc(lwc_source)
    cloud_obj = LwcStatus(lwc_source, lwc_obj)
    error_obj = LwcError(lwc_source, lwc_obj)
    _append_data(lwc_source, lwc_obj, cloud_obj, error_obj)
    date = lwc_source.get_date()
    attributes = output.add_time_attribute(LWC_ATTRIBUTES, date)
    output.update_attributes(lwc_source.data, attributes)
    uuid = output.save_product_file('lwc', lwc_source, output_file, keep_uuid, uuid,
                                    copy_from_cat=('lwp', 'lwp_error'))
    lwc_source.close()
    return uuid


class LwcSource(DataSource):
    """Data container for liquid water content calculations. Child of DataSource.

    This class reads input data from a categorize file and provides data
    structures and methods for holding the results.

    Args:
        categorize_file (str): Categorize file name.

    Attributes:
        lwp (ndarray): 1D liquid water path.
        lwp_error (ndarray): 1D error of liquid water path.
        is_rain (ndarray): 1D array denoting presence of rain.
        dheight (float): Median difference in height vector.
        atmosphere (dict): Dictionary containing interpolated fields `temperature`
            and `pressure`.
        categorize_bits (CategorizeBits): The :class:`CategorizeBits` instance.

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
    """Class handling the actual LWC calculations.

    Args:
        lwc_source (LwcSource): The :class:`LwcSource` instance.

    Attributes:
        lwc_source (LwcSource): The :class:`LwcSource` instance.
        dheight (float): Median difference in height vector.
        is_liquid (ndarray): 2D array denoting liquid.
        lwc_adiabatic (ndarray): 2D array storing adiabatic lwc.
        lwc (ndarray): 2D array of liquid water content (scaled with lwp).

    """
    def __init__(self, lwc_source):
        self.lwc_source = lwc_source
        self.dheight = self.lwc_source.dheight
        self.is_liquid = self._get_liquid()
        self.lwc_adiabatic = self._init_lwc_adiabatic()
        self.lwc = self._adiabatic_lwc_to_lwc()
        self.screen_rain()

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

    def screen_rain(self):
        """Masks profiles with rain."""
        is_rain = self.lwc_source.is_rain.astype(bool)
        self.lwc[is_rain, :] = ma.masked


class LwcStatus:
    """Adjust clouds (where possible) so that theoretical and measured LWP agree.

    Args:
        lwc_source (LwcSource): The :class:`LwcSource` instance.
        lwc_obj (Lwc):  The :class:`Lwc` instance.

    Attributes:
        lwc_source (LwcSource): The :class:`LwcSource` instance.
        lwc (Lwc): The :class:`Lwc` instance
        is_liquid (ndarray): 2D array denoting liquid.
        lwc_adiabatic (ndarray): 2D array storing adiabatic lwc.
        echo (dict): Dictionary storing radar and lidar echos
        status (ndarray): 2D array storing lwc status classification

    """
    def __init__(self, lwc_source, lwc_obj):
        self.lwc_source = lwc_source
        self.lwc = lwc_obj.lwc
        self.is_liquid = lwc_obj.is_liquid
        self.lwc_adiabatic = lwc_obj.lwc_adiabatic
        self.echo = self._get_echo()
        self.status = self._init_status()
        self._adjust_cloud_tops(self._find_adjustable_clouds())
        self.screen_rain()

    def _init_status(self):
        status = ma.zeros(self.is_liquid.shape, dtype=int)
        status[self.is_liquid] = 1
        return status

    def _get_echo(self):
        quality_bits = self.lwc_source.categorize_bits.quality_bits
        return {'radar': quality_bits['radar'], 'lidar': quality_bits['lidar']}

    def _adjust_cloud_tops(self, adjustable_clouds):
        """Adjusts cloud top index so that measured lwc corresponds to
        theoretical value.
        """
        for time_index in np.unique(np.where(adjustable_clouds)[0]):
            base_index = np.where(adjustable_clouds[time_index, :])[0][0]
            self._update_status(time_index)
            self._adjust_lwc(time_index, base_index)

    def _update_status(self, time_ind):
        alt_indices = np.where(self.is_liquid[time_ind, :])[0]
        self.status[time_ind, alt_indices] = 2

    def _adjust_lwc(self, time_ind, base_ind):
        lwc_base = self.lwc_adiabatic[time_ind, base_ind]
        distance_from_base = 1
        while True:
            top_ind = base_ind + distance_from_base
            lwc_top = lwc_base * (distance_from_base + 1)
            self.lwc_adiabatic[time_ind, top_ind] = lwc_top
            if not self.status[time_ind, top_ind]:
                self.status[time_ind, top_ind] = 3
            if self._has_converged(time_ind) or self._out_of_bound(top_ind):
                break
            distance_from_base += 1

    def _has_converged(self, ind):
        lwc_sum = ma.sum(self.lwc_adiabatic[ind, :])
        if lwc_sum * self.lwc_source.dheight > self.lwc_source.lwp[ind]:
            return True
        return False

    def _out_of_bound(self, ind):
        return ind >= self.lwc.shape[1] - 1

    def _find_adjustable_clouds(self):
        top_clouds = self._find_topmost_clouds()
        detection_type = self._find_echo_combinations_in_liquid()
        detection_type[~top_clouds] = 0
        lidar_only_clouds = self._find_lidar_only_clouds(detection_type)
        top_clouds[~lidar_only_clouds, :] = 0
        top_clouds = self._remove_good_profiles(top_clouds)
        return top_clouds

    def _find_topmost_clouds(self):
        top_clouds = np.copy(self.is_liquid)
        cloud_edges = top_clouds[:, :-1][:, ::-1] < top_clouds[:, 1:][:, ::-1]
        topmost_bases = self.is_liquid.shape[1] - 1 - np.argmax(cloud_edges, axis=1)
        for n, base in enumerate(topmost_bases):
            top_clouds[n, :base] = 0
        return top_clouds

    def _find_echo_combinations_in_liquid(self):
        """Classifies liquid clouds by detection type: 1=lidar, 2=radar, 3=both."""
        lidar_detected = (self.is_liquid & self.echo['lidar']).astype(int)
        radar_detected = (self.is_liquid & self.echo['radar']).astype(int) * 2
        return lidar_detected + radar_detected

    @staticmethod
    def _find_lidar_only_clouds(detection):
        """Finds top clouds that contain only lidar-detected pixels.

        Args:
            detection (ndarray): Array of integers where 1=lidar, 2=radar, 3=both.

        Returns:
            ndarray: Boolean array containing top-clouds that are detected only
            by lidar.

        """
        sum_of_cloud_pixels = ma.sum(detection > 0, axis=1)
        sum_of_detection_type = ma.sum(detection, axis=1)
        return sum_of_cloud_pixels / sum_of_detection_type == 1

    def _remove_good_profiles(self, top_clouds):
        no_rain = ~self.lwc_source.is_rain.astype(bool)
        lwp_difference = self._find_lwp_difference()
        dubious_profiles = (lwp_difference < 0) & no_rain
        top_clouds[~dubious_profiles, :] = 0
        return top_clouds

    def _find_lwp_difference(self):
        """Returns difference of theoretical LWP and measured LWP (g/m2).

        In theory, this difference should be always positive. Negative values
        indicate missing (or too narrow) liquid clouds.
        """
        lwc_sum = ma.sum(self.lwc_adiabatic, axis=1) * self.lwc_source.dheight
        return lwc_sum - self.lwc_source.lwp

    def screen_rain(self):
        """Masks profiles with rain."""
        is_rain = self.lwc_source.is_rain.astype(bool)
        self.status[is_rain, :] = 4


class LwcError:
    """Calculates liquid water content error.

    Args:
        lwc_source (LwcSource): The :class:`LwcSource` instance.
        lwc_obj (Lwc):  The :class:`Lwc` instance.

    Attributes:
        lwc_source (LwcSource): The :class:`LwcSource` instance.
        lwc (Lwc): The :class:`Lwc` instance
        lwc_error (ndarray): 2D array storing lwc_error.

    """
    def __init__(self, lwc_source, lwc_obj):
        self.lwc = lwc_obj.lwc
        self.lwc_source = lwc_source
        self.lwc_error = self._calculate_lwc_error()
        self.screen_rain()

    def _calculate_lwc_error(self):
        lwc_relative_error = self._calc_lwc_relative_error()
        lwp_relative_error = self._calc_lwp_relative_error()
        combined_error = self._calc_combined_error(lwc_relative_error, lwp_relative_error)
        return self._fill_error_array(combined_error)

    @staticmethod
    def _limit_error(error, max_value):
        error[error > max_value] = max_value
        return error

    def _calc_lwc_gradient(self):
        gradient_elements = np.gradient(self.lwc.filled(0))
        return utils.l2norm(*gradient_elements)

    def _calc_lwc_relative_error(self):
        lwc_gradient = self._calc_lwc_gradient()
        error = lwc_gradient / self.lwc / 2
        return self._limit_error(error, 5)

    def _calc_lwp_relative_error(self):
        err = self.lwc_source.lwp_error
        value = self.lwc_source.lwp
        error = np.divide(err, value, out=np.zeros_like(err), where=value != 0)
        return self._limit_error(error, 10)

    @staticmethod
    def _calc_combined_error(error_2d, error_1d):
        error_1d_transposed = utils.transpose(error_1d)
        return utils.l2norm(error_2d, error_1d_transposed)

    def _fill_error_array(self, error_in):
        lwc_error = ma.masked_all(self.lwc.shape)
        ind = ma.where(self.lwc)
        lwc_error[ind] = error_in[ind]
        return lwc_error

    def screen_rain(self):
        """Masks profiles with rain."""
        is_rain = self.lwc_source.is_rain.astype(bool)
        self.lwc_error[is_rain, :] = ma.masked


def _append_data(lwc_data, lwc_obj, cloud_obj, error_obj):
    lwc_data.append_data(lwc_obj.lwc * G_TO_KG, 'lwc', units='kg m-3')
    lwc_data.append_data(cloud_obj.status, 'lwc_retrieval_status')
    lwc_data.append_data(error_obj.lwc_error, 'lwc_error', units='dB')


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
         'retrieval and the uncertainty in cloud base and/or cloud top height.'),

    'lwc_retrieval_status':
        ('This variable describes whether a retrieval was performed for each\n'
         'pixel, and its associated quality.')
}

DEFINITIONS = {
    'lwc_retrieval_status':
        ('\n'
         'Value 0: No liquid water detected.\n'
         'Value 1: Reliable retrieval.\n'
         'Value 2: Cloud pixel whose top has been adjusted so that the theoretical\n'
         '         liquid water path would match observation.\n'
         'Value 3: New cloud pixel introduced so that the theoretical liquid\n'
         '         water path would match observation.\n'
         'Value 4: Rain present: cloud extent is difficult to ascertain and liquid\n'
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
        units='dB'
    ),
    'lwc_retrieval_status': MetaData(
        long_name='Liquid water content retrieval status',
        comment=COMMENTS['lwc_retrieval_status'],
        definition=DEFINITIONS['lwc_retrieval_status']
    ),
}
