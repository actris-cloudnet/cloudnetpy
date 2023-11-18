"""Mwr module, containing the :class:`Mwr` class."""
import numpy as np

from cloudnetpy import utils
from cloudnetpy.constants import G_TO_KG
from cloudnetpy.datasource import DataSource


class Mwr(DataSource):
    """Microwave radiometer class, child of DataSource.

    Args:
    ----
         full_path: Cloudnet Level 1b mwr file.

    """

    def __init__(self, full_path: str):
        super().__init__(full_path)
        self._init_lwp_data()
        self._init_lwp_error()

    def rebin_to_grid(self, time_grid: np.ndarray) -> None:
        """Approximates lwp and its error in a grid using mean.

        Args:
        ----
            time_grid: 1D target time grid.

        """
        for array in self.data.values():
            array.rebin_data(self.time, time_grid)

    def _init_lwp_data(self) -> None:
        lwp = self.dataset.variables["lwp"][:]
        self.append_data(lwp, "lwp")

    def _init_lwp_error(self) -> None:
        random_error, bias = 0.25, 20
        lwp_error = utils.l2norm(self.data["lwp"][:] * random_error, bias * G_TO_KG)
        self.append_data(lwp_error, "lwp_error", units="kg m-2")
        self.data["lwp_error"].comment = (
            "This variable is a rough estimate of the one-standard-deviation\n"
            f"error in liquid water path, calculated as a combination of\n"
            f"a {bias} g m-2 linear error and a {round(random_error*100)} %\n"
            "fractional error."
        )
