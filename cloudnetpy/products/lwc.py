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
        self.is_liquid = self.category_bits['droplet']
        self.liquid_bases = atmos.find_cloud_bases(self.is_liquid)


class Lwc:
    """Class handling the liquid water content calculations."""
    def __init__(self, lwc_input, liquid):
        self.lwc_input = lwc_input
        self.liquid = liquid
        self.lwc_adiabatic = None
        self.lwc_scaled = self._get_lwc()
        self.melting_in_profile = np.any(liquid.category_bits['melting'], axis=1)

    def _check_suspicious_profiles(self):
        lwc_sum = ma.sum(self.lwc_adiabatic, axis=1)
        lwp = self.lwc_input.lwp
        bad_indices = (self.melting_in_profile.astype(bool)
                       | self.lwc_input.rain_in_profile.astype(bool)
                       | (lwp < 0))
        lwc_sum[bad_indices] = ma.masked
        lwp[bad_indices] = ma.masked
        #plt.plot(lwc_sum*self.lwc_input.dheight/1000, 'r.')
        #plt.plot(lwp/1000, 'b.')
        #plt.ylim(-0.2, 5)
        #plt.show()

    def _get_lwc(self):
        lwp = self.lwc_input.lwp
        atmosphere = (self.lwc_input.temperature, self.lwc_input.pressure)
        is_liquid = self.liquid.is_liquid
        dheight = self.lwc_input.dheight
        lwc_change_rate = atmos.fill_clouds_with_lwc_dz(atmosphere, is_liquid)
        lwc = atmos.calc_adiabatic_lwc(lwc_change_rate, is_liquid, dheight)
        self.lwc_adiabatic = lwc

        self._check_suspicious_profiles(self)

        lwc_norm = atmos.scale_lwc(lwc, lwp) * G_TO_KG
        return lwc_norm


def generate_lwc(categorize_file):
    """High level API to generate Cloudnet liquid water content file."""
    lwc_input = LwcSource(categorize_file)
    liquid = Liquid(lwc_input)
    lwc = Lwc(lwc_input, liquid)

