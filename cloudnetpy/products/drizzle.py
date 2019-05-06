"""Module for creating Cloudnet drizzle product.
"""
import numpy as np
import numpy.ma as ma
from cloudnetpy import utils
from cloudnetpy.categorize import DataSource
from cloudnetpy.products import product_tools as p_tools
from cloudnetpy.products.product_tools import ProductClassification
from cloudnetpy.plotting import plot_2d


def generate_drizzle(categorize_file, output_file):
    drizze_data = DrizzleSource(categorize_file)
    drizzle_class = DrizzleClassification(categorize_file)


class DrizzleSource(DataSource):
    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.radar_frequency = self.getvar('radar_frequency')
        self.wl_band = utils.get_wl_band(self.radar_frequency)


class DrizzleClassification(ProductClassification):
    """Class storing the information about different drizzle types."""

    def __init__(self, categorize_file):
        super().__init__(categorize_file)
        self.warm_liquid = self._find_warm_liquid()
        self.drizzle = self._find_drizzle()
        self.would_be_drizzle = self._find_would_be_drizzle()
        self.cold_rain = self._find_cold_rain()

    def _find_warm_liquid(self):
        return (self.category_bits['droplet']
                & ~self.category_bits['cold'])

    def _find_drizzle(self):
        return (~utils.transpose(self.is_rain)
                & self.category_bits['falling']
                & ~self.category_bits['droplet']
                & ~self.category_bits['cold']
                & ~self.category_bits['melting']
                & ~self.category_bits['insect']
                & self.quality_bits['radar']
                & self.quality_bits['lidar']
                & ~self.quality_bits['clutter']
                & ~self.quality_bits['molecular']
                & ~self.quality_bits['attenuated'])

    def _find_would_be_drizzle(self):
        return (~utils.transpose(self.is_rain)
                & self.warm_liquid
                & self.category_bits['falling']
                & ~self.category_bits['melting']
                & ~self.category_bits['insect']
                & self.quality_bits['radar']
                & ~self.quality_bits['clutter']
                & ~self.quality_bits['molecular'])

    def _find_cold_rain(self):
        return np.any(self.category_bits['melting'], axis=1)
