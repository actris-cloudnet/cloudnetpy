"""Module with a class for Lufft chm15k ceilometer."""
import numpy as np
from cloudnetpy.instruments.ceilometer import Ceilometer, NoiseParam
from cloudnetpy import utils
from typing import Optional
import logging


class NcLidar(Ceilometer):
    """Class for all lidars using netCDF files."""

    def __init__(self, noise_param: NoiseParam):
        super().__init__(noise_param)
        self.dataset = None

    def _fetch_range(self, reference: str) -> None:
        range_instrument = self.dataset.variables['range'][:]
        self.data['range'] = utils.edges2mid(range_instrument, reference)

    def _fetch_time_and_date(self) -> None:
        time = self.dataset.variables['time']
        self.data['time'] = time[:]
        epoch = utils.get_epoch(time.units)
        self.get_date_and_time(epoch)

    def _fetch_tilt_angle(self, key: str, default: Optional[float] = 3) -> None:
        if key in self.dataset.variables:
            tilt_angle = self.dataset.variables[key][:]
        else:
            tilt_angle = default
            logging.warning(f'No tilt angle found, assuming {tilt_angle} degrees.')
        self.data['tilt_angle'] = np.array(tilt_angle)
