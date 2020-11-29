"""Mwr module, containing the :class:`Mwr` class."""
from cloudnetpy.categorize import DataSource
from cloudnetpy import utils


class Mwr(DataSource):
    """Microwave radiometer class, child of DataSource.

    Args:
         mwr_file (str): File name of the calibrated mwr file.

    """
    def __init__(self, mwr_file):
        super().__init__(mwr_file)
        self._init_lwp_data()
        self._init_lwp_error()

    def rebin_to_grid(self, time_grid):
        """Rebinning of lwp and its error.

        Args:
            time_grid (ndarray): 1D target time grid.

        """
        for key in self.data:
            self.data[key].rebin_data(self.time, time_grid)

    def _init_lwp_data(self):
        possible_names = ('LWP_data', 'lwp', 'LWP', 'clwvi')
        for name in possible_names:
            if name in self.dataset.variables:
                lwp = self.dataset.variables[name][:]
                unit = self.dataset.variables[name].units
        if 'kg' in unit:
            lwp *= 1000
        lwp[lwp < 0] = 0
        self.append_data(lwp, 'lwp', units='g m-2')

    def _init_lwp_error(self):
        # TODO: Check these error values
        random_error, bias = 0.25, 50
        lwp_error = utils.l2norm(self.data['lwp'][:]*random_error, bias)
        self.append_data(lwp_error, 'lwp_error', units='g m-2')
