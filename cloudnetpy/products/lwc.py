import numpy as np
import numpy.ma as ma
from cloudnetpy.categorize import DataSource
from cloudnetpy import utils
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy import plotting
from cloudnetpy import atmos


class LwcSource(DataSource):
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.temperature = self._interpolate_model_field('temperature')
        self.pressure = self._interpolate_model_field('pressure')
        self.lwp = self.getvar('lwp')
        self.lwp_error = self.getvar('lwp_error')
        self.dheight = utils.mdiff(self.getvar('height'))

    def _interpolate_model_field(self, variable_name):
        """Interpolates 2D model field into Cloudnet grid."""
        return utils.interpolate_2d(self.getvar('model_time'),
                                    self.getvar('model_height'),
                                    self.getvar(variable_name),
                                    self.time, self.getvar('height'))


class Liquid:
    def __init__(self, categorize_object):
        self._category_bits = p_tools.read_category_bits(categorize_object)
        self.is_liquid = self._category_bits['droplet']
        self.liquid_bases = self._find_liquid_bases()

    def _find_liquid_bases(self):
        liquid_bases = np.zeros_like(self.is_liquid, dtype=bool)
        for ind, profile in enumerate(self.is_liquid):
            bases, _ = utils.bases_and_tops(profile)
            liquid_bases[ind, bases] = 1
        return liquid_bases


class Lwc:
    def __init__(self, lwc_input, liquid):
        self.lwc_input = lwc_input
        self.liquid = liquid
        self.lwc = self._get_lwc()

    def _get_lwc(self):
        lwp = self.lwc_input.lwp
        atmosphere = (self.lwc_input.temperature, self.lwc_input.pressure)
        is_liquid = self.liquid.is_liquid
        dheight = self.lwc_input.dheight
        lwc_change_rate = atmos.fill_clouds_with_lwc_dz(atmosphere, is_liquid)
        lwc = atmos.calc_adiabatic_lwc(lwc_change_rate, is_liquid, dheight)
        lwc_norm = atmos.normalize_lwc(lwc, lwp)

        plotting.plot_2d(lwc)
        #plotting.plot_2d(lwc_norm)

        return lwc_norm


def generate_lwc(categorize_file):
    lwc_input = LwcSource(categorize_file)
    liquid = Liquid(lwc_input)
    lwc = Lwc(lwc_input, liquid)

