"""Mwr module, containing the :class:`Mwr` class."""
import numpy as np
from cloudnetpy.categorize import DataSource
from cloudnetpy import utils


class Mwr(DataSource):
    """Microwave radiometer class, child of DataSource.

    Args:
         full_path: Cloudnet Level 1b mwr file.

    """
    def __init__(self, full_path: str):
        super().__init__(full_path)
        self._init_lwp_data()
        self._init_lwp_error()

    def rebin_to_grid(self, time_grid: np.ndarray) -> None:
        """Approximates lwp and its error in a grid using mean.

        Args:
            time_grid: 1D target time grid.

        """
        for key in self.data:
            self.data[key].rebin_data(self.time, time_grid)

    def _init_lwp_data(self) -> None:
        lwp, unit = None, None
        possible_names = ('LWP_data', 'lwp', 'LWP', 'clwvi')
        for name in possible_names:
            if name in self.dataset.variables:
                lwp = self.dataset.variables[name][:]
                unit = self.dataset.variables[name].units
        if lwp is None or unit is None:
            raise RuntimeError('Error: Can not find LWP or determine its unit.')
        if 'kg' in unit:
            lwp *= 1000
        lwp[lwp < 0] = 0
        self.append_data(lwp, 'lwp', units='g m-2')

    def _init_lwp_error(self) -> None:
        # TODO: Check these error values
        random_error, bias = 0.25, 50
        lwp_error = utils.l2norm(self.data['lwp'][:]*random_error, bias)
        self.append_data(lwp_error, 'lwp_error', units='g m-2')
