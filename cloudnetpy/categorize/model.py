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
        model_file: File name of the NWP model file.
        alt_site: Altitude of the site above mean sea level (m).

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

    def __init__(self, model_file: str, alt_site: float):
        super().__init__(model_file)
        self.type = _find_model_type(model_file)
        self.model_heights = self._get_model_heights(alt_site)
        self.mean_height = _calc_mean_height(self.model_heights)
        self.height = None
        self.data_sparse = {}
        self.data_dense = {}
        self._append_grid()

    def interpolate_to_common_height(self, wl_band: int) -> None:
        """Interpolates model variables to common height grid.

        Args:
            wl_band: Integer denoting the approximate wavelength band of the
                cloud radar (0 = ~35.5 GHz, 1 = ~94 GHz).

        """
        def _interpolate_variable(data_in: ma.MaskedArray) -> CloudnetArray:
            datai = ma.zeros((len(self.time), len(self.mean_height)))
            for ind, (alt, prof) in enumerate(zip(self.model_heights, data_in)):
                if prof.mask.all():
                    datai[ind, :] = ma.masked
                else:
                    fun = interp1d(alt, prof, fill_value='extrapolate')
                    datai[ind, :] = fun(self.mean_height)
            return CloudnetArray(datai, key, units)

        for key in self.fields_sparse:
            variable = self.dataset.variables[key]
            data = variable[:]
            units = variable.units
            if 'atten' in key:
                data = data[wl_band, :, :]
            self.data_sparse[key] = _interpolate_variable(data)

    def interpolate_to_grid(self, time_grid: np.ndarray, height_grid: np.ndarray) -> None:
        """Interpolates model variables to Cloudnet's dense time / height grid.

        Args:
            time_grid: The target time array (fraction hour).
            height_grid: The target height array (m).

        """
        for key in self.fields_dense:
            array = self.data_sparse[key][:]
            valid_profiles = _find_number_of_valid_profiles(array)
            if valid_profiles < 2:
                raise RuntimeError('Bad model file: too few proper profiles')
            self.data_dense[key] = utils.interpolate_2d_mask(self.time,
                                                             self.mean_height,
                                                             array,
                                                             time_grid, height_grid)
        self.height = height_grid

    def calc_wet_bulb(self) -> None:
        """Calculates wet-bulb temperature in dense grid."""
        wet_bulb_temp = atmos.calc_wet_bulb_temperature(self.data_dense)
        self.append_data(wet_bulb_temp, 'Tw', units='K')

    def screen_sparse_fields(self) -> None:
        """Removes model fields that we don't want to write in the output."""
        fields_to_keep = ('temperature', 'pressure', 'q', 'uwind', 'vwind')
        self.data_sparse = {key: self.data_sparse[key] for key in fields_to_keep}

    def _append_grid(self) -> None:
        self.append_data(np.array(self.time), 'model_time')
        self.append_data(self.mean_height, 'model_height')

    def _get_model_heights(self, alt_site: float) -> np.ndarray:
        """Returns model heights for each time step."""
        model_heights = self.dataset.variables['height']
        return self.km2m(model_heights) + alt_site


def _calc_mean_height(model_heights: np.ndarray) -> np.ndarray:
    mean_height = ma.mean(model_heights, axis=0)
    return np.array(mean_height)


def _find_model_type(file_name: str) -> str:
    """Finds model type from the model filename."""
    possible_keys = utils.fetch_cloudnet_model_types()
    for key in possible_keys:
        if key in file_name:
            return key
    raise ValueError('Unknown model type')


def _find_number_of_valid_profiles(array: np.ndarray) -> int:
    n_good = 0
    for row in array:
        if not hasattr(row, 'mask') or np.sum(row.mask.astype(int)) == 0:
            n_good += 1
    return n_good
