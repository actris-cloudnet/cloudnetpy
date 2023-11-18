"""Module with a class for Lufft chm15k ceilometer."""
import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy import ma

from cloudnetpy import utils
from cloudnetpy.instruments.ceilometer import Ceilometer

if TYPE_CHECKING:
    import netCDF4


class NcLidar(Ceilometer):
    """Class for all lidars using netCDF files."""

    def __init__(self):
        super().__init__()
        self.dataset: netCDF4.Dataset | None = None

    def _fetch_range(self, reference: Literal["upper", "lower"]) -> None:
        if self.dataset is None:
            msg = "No dataset found"
            raise RuntimeError(msg)
        range_instrument = self.dataset.variables["range"][:]
        self.data["range"] = utils.edges2mid(range_instrument, reference)

    def _fetch_time_and_date(self) -> None:
        if self.dataset is None:
            msg = "No dataset found"
            raise RuntimeError(msg)
        time = self.dataset.variables["time"]
        self.data["time"] = time[:]
        epoch = utils.get_epoch(time.units)
        self.get_date_and_time(epoch)

    def _fetch_zenith_angle(self, key: str, default: float = 3.0) -> None:
        if self.dataset is None:
            msg = "No dataset found"
            raise RuntimeError(msg)
        if key in self.dataset.variables:
            zenith_angle = ma.median(self.dataset.variables[key][:])
        else:
            zenith_angle = float(default)
            logging.warning("No zenith angle found, assuming %s degrees", zenith_angle)
        if zenith_angle == 0:
            logging.warning("Zenith angle 0 degrees - risk of specular reflection")
        self.data["zenith_angle"] = np.array(zenith_angle)
