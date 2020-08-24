"""Model module, containing the :class:`Model` class."""
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
from cloudnetpy import CloudnetArray
from cloudnetpy.categorize import DataSource, atmos
from cloudnetpy import utils


class Model(DataSource):
    """Model class, child of DataSource.

    Args:
        model_file (str): File name of the NWP model file.
        alt_site (float): Altitude of the site above mean sea level (m).

    Attributes:
        type (str): Model type, e.g. 'gdas1' or 'ecwmf'.
        model_heights (ndarray): 2-D array of model heights (one for each time
            step).
        mean_height (ndarray): Mean of *model_heights*.
        data_sparse (dict): Model variables in common height grid but without
            interpolation in time.
        data_dense (dict): Model variables interpolated to Cloudnet's dense
            time / height grid.

    """
    fields_dense = ('temperature', 'pressure', 'rh',
                    'gas_atten', 'specific_gas_atten',
                    'specific_saturated_gas_atten',
                    'specific_liquid_atten')
    fields_sparse = fields_dense + ('q', 'uwind', 'vwind')

    def __init__(self, model_file, alt_site):
        super().__init__(model_file)
        self.type = _find_model_type(model_file)
        self.model_heights = self._get_model_heights(alt_site)
        self.mean_height = _calc_mean_height(self.model_heights)
        self.height = None
        self.data_sparse = {}
        self.data_dense = {}
        self._append_grid()

    def interpolate_to_common_height(self, wl_band):
        """Interpolates model variables to common height grid.

        Args:
            wl_band (int): Integer denoting the approximate wavelength
                band of the cloud radar (0 = ~35.5 GHz, 1 = ~94 GHz).

        """
        def _interpolate_variable():
            datai = np.zeros((len(self.time), len(self.mean_height)))
            for ind, (alt, prof) in enumerate(zip(self.model_heights, data)):
                fun = interp1d(alt, prof, fill_value='extrapolate')
                datai[ind, :] = fun(self.mean_height)
            return CloudnetArray(datai, key, units)

        for key in self.fields_sparse:
            variable = self.dataset.variables[key]
            data = np.array(variable[:])
            units = variable.units
            if 'atten' in key:
                data = data[wl_band, :, :]
            self.data_sparse[key] = _interpolate_variable()

    def interpolate_to_grid(self, time_grid, height_grid):
        """Interpolates model variables to Cloudnet's dense time / height grid.

        Args:
            time_grid (ndarray): The target time array (fraction hour).
            height_grid (ndarray): The target height array (m).

        """
        for key in self.fields_dense:
            self.data_dense[key] = utils.interpolate_2d(self.time,
                                                        self.mean_height,
                                                        self.data_sparse[key][:],
                                                        time_grid, height_grid)
        self.height = height_grid

    def calc_wet_bulb(self):
        """Calculates wet-bulb temperature in dense grid."""
        wet_bulb_temp = atmos.calc_wet_bulb_temperature(self.data_dense)
        self.append_data(wet_bulb_temp, 'Tw', units='K')

    def screen_sparse_fields(self):
        """Removes model fields that we don't want to write in the output."""
        fields_to_keep = ('temperature', 'pressure', 'q', 'uwind', 'vwind')
        self.data_sparse = {key: self.data_sparse[key]
                            for key in fields_to_keep}

    def _append_grid(self):
        self.append_data(self.time, 'model_time')
        self.append_data(self.mean_height, 'model_height')

    def _get_model_heights(self, alt_site):
        """Returns model heights for each time step."""
        model_heights = self.dataset.variables['height']
        if ma.count_masked(model_heights[:] > 0):
            raise RuntimeError('Masked values in the data file! Aborting..')
        return self.km2m(model_heights) + alt_site


def _calc_mean_height(model_heights):
    return np.mean(np.array(model_heights), axis=0)


def _find_model_type(file_name):
    """Finds model type from the model filename."""
    possible_keys = ('ecmwf', 'gdas')
    for key in possible_keys:
        if key in file_name:
            return key
    return ''
