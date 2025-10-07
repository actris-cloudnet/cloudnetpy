"""Mwr module, containing the :class:`Mwr` class."""

from os import PathLike

import numpy.typing as npt

from cloudnetpy.datasource import DataSource
from cloudnetpy.exceptions import DisdrometerDataError
from cloudnetpy.utils import interpolate_1d


class Disdrometer(DataSource):
    """Disdrometer class, child of DataSource.

    Args:
    ----
         full_path: Cloudnet Level 1b disdrometer file.

    """

    def __init__(self, full_path: str | PathLike) -> None:
        super().__init__(full_path)
        self._init_rainfall_rate()

    def interpolate_to_grid(self, time_grid: npt.NDArray) -> None:
        for key, array in self.data.items():
            method = "nearest" if key == "synop_WaWa" else "linear"
            self.data[key].data = interpolate_1d(
                self.time, array.data, time_grid, max_time=1, method=method
            )

    def _init_rainfall_rate(self) -> None:
        keys = ("rainfall_rate", "n_particles", "synop_WaWa")
        for key in keys:
            if key not in self.dataset.variables:
                if key == "synop_WaWa":
                    continue
                msg = f"variable {key} is missing"
                raise DisdrometerDataError(msg)
            self.append_data(self.dataset.variables[key][:], key)
