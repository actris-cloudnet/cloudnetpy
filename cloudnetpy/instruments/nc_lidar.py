"""Module with a class for Lufft chm15k ceilometer."""
import logging

import netCDF4
import numpy as np

from cloudnetpy import utils
from cloudnetpy.instruments.ceilometer import Ceilometer


class NcLidar(Ceilometer):
    """Class for all lidars using netCDF files."""

    def __init__(self):
        super().__init__()
        self.dataset: netCDF4.Dataset = None

    def _fetch_range(self, reference: str) -> None:
        range_instrument = self.dataset.variables["range"][:]
        self.data["range"] = utils.edges2mid(range_instrument, reference)

    def _fetch_time_and_date(self) -> None:
        time = self.dataset.variables["time"]
        self.data["time"] = time[:]
        epoch = utils.get_epoch(time.units)
        self.get_date_and_time(epoch)

    def _fetch_zenith_angle(self, key: str, default: float = 3.0) -> None:
        if key in self.dataset.variables:
            zenith_angle = self.dataset.variables[key][:]
        else:
            zenith_angle = float(default)
            logging.warning(f"No zenith angle found, assuming {zenith_angle} degrees")
        if zenith_angle == 0:
            logging.warning("Zenith angle 0 degrees - risk of specular reflection")
        self.data["zenith_angle"] = np.array(zenith_angle)
