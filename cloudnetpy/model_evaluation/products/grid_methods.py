import numpy as np
from numpy import ma

from cloudnetpy import utils
from cloudnetpy.model_evaluation.products import tools as tl
from cloudnetpy.model_evaluation.products.model_products import ModelManager
from cloudnetpy.model_evaluation.products.observation_products import ObservationManager


class ProductGrid:
    """Class to generate downsampling of observation product to model grid.

    Args:
        model_obj (object): The :class:'ModelManager' object.
        obs_obj (object): The :class:'ObservationManager' object.

    Notes:
        Downsampled observation products data is added to a ModelManager
        object which is used for nc-file creation and writing
    """

    def __init__(self, model_obj: ModelManager, obs_obj: ObservationManager):
        self._obs_obj = obs_obj
        self._date = obs_obj.date
        self._obs_time = tl.time2datetime(obs_obj.time, self._date)
        self._obs_height = obs_obj.data["height"][:]
        self._obs_data = obs_obj.data[obs_obj.obs][:]
        self.model_obj = model_obj
        self._model_time = model_obj.time
        self._model_height = model_obj.data[model_obj.keys["height"]][:]
        self._time_adv = tl.calculate_advection_time(
            int(model_obj.resolution_h),
            ma.array(model_obj.wind),
            1,
        )
        time_steps = utils.binvec(self._model_time)
        self._time_steps = tl.time2datetime(time_steps, self._date)
        self._generate_downsample_product()

    def _generate_downsample_product(self) -> None:
        """Downsampling products are generated with different averaging methods
        for a selected size of model time-height window.
        """
        product_dict, product_adv_dict = self._get_method_storage()
        model_t = tl.time2datetime(self._model_time, self._date)
        for i in range(len(self._time_steps) - 1):
            x_ind = tl.get_1d_indices(
                (self._time_steps[i], self._time_steps[i + 1]),
                self._obs_time,
            )
            if self._obs_obj.obs == "iwc":
                x_ind_no_rain = tl.get_1d_indices(
                    (self._time_steps[i], self._time_steps[i + 1]),
                    self._obs_time,
                    mask=self._obs_obj.data["iwc_rain"][:],
                )
            y_steps = tl.rebin_edges(self._model_height[i])
            for j in range(len(y_steps) - 1):
                x_ind_adv = tl.get_adv_indices(
                    model_t[i],
                    self._time_adv[i, j],
                    self._obs_time,
                )
                y_ind = tl.get_1d_indices(
                    (y_steps[j], y_steps[j + 1]),
                    self._obs_height,
                )
                ind = np.outer(x_ind, y_ind)
                ind_avd = np.outer(x_ind_adv, y_ind)
                if self._obs_obj.obs == "cf":
                    data = self._reshape_data_to_window(ind, x_ind, y_ind)
                    if data is None:
                        continue
                    product_dict = self._regrid_cf(product_dict, i, j, data)
                    data_adv = self._reshape_data_to_window(ind_avd, x_ind_adv, y_ind)
                    if data_adv is None:
                        msg = "No data for advection"
                        raise RuntimeError(msg)
                    product_adv_dict = self._regrid_cf(product_adv_dict, i, j, data_adv)
                elif self._obs_obj.obs == "iwc":
                    x_ind_no_rain_adv = tl.get_adv_indices(
                        model_t[i],
                        self._time_adv[i, j],
                        self._obs_time,
                        mask=self._obs_obj.data["iwc_rain"][:],
                    )
                    ind_no_rain = np.outer(x_ind_no_rain, y_ind)
                    ind_no_rain_adv = np.outer(x_ind_no_rain_adv, y_ind)
                    product_dict = self._regrid_iwc(
                        product_dict,
                        i,
                        j,
                        ind,
                        ind_no_rain,
                    )
                    product_adv_dict = self._regrid_iwc(
                        product_adv_dict,
                        i,
                        j,
                        ind_avd,
                        ind_no_rain_adv,
                    )
                else:
                    product_dict = self._regrid_product(product_dict, i, j, ind)
                    product_adv_dict = self._regrid_product(
                        product_adv_dict,
                        i,
                        j,
                        ind_avd,
                    )
        self._append_data2object([product_dict, product_adv_dict])

    def _get_method_storage(self) -> tuple[dict, dict]:
        if self._obs_obj.obs == "cf":
            return self._cf_method_storage()
        if self._obs_obj.obs == "iwc":
            return self._iwc_method_storage()
        return self._product_method_storage()

    def _cf_method_storage(self) -> tuple[dict, dict]:
        cf_dict = {
            "cf_V": ma.zeros(self._model_height.shape),
            "cf_A": ma.zeros(self._model_height.shape),
        }
        cf_adv_dict = {
            "cf_V_adv": ma.zeros(self._model_height.shape),
            "cf_A_adv": ma.zeros(self._model_height.shape),
        }
        return cf_dict, cf_adv_dict

    def _iwc_method_storage(self) -> tuple[dict, dict]:
        iwc_dict = {
            "iwc": ma.zeros(self._model_height.shape),
            "iwc_att": ma.zeros(self._model_height.shape),
            "iwc_rain": ma.zeros(self._model_height.shape),
        }
        iwc_adv_dict = {
            "iwc_adv": ma.zeros(self._model_height.shape),
            "iwc_att_adv": ma.zeros(self._model_height.shape),
            "iwc_rain_adv": ma.zeros(self._model_height.shape),
        }
        return iwc_dict, iwc_adv_dict

    def _product_method_storage(self) -> tuple[dict, dict]:
        product_dict = {f"{self._obs_obj.obs}": ma.zeros(self._model_height.shape)}
        product_adv_dict = {
            f"{self._obs_obj.obs}_adv": ma.zeros(self._model_height.shape),
        }
        return product_dict, product_adv_dict

    def _reshape_data_to_window(
        self,
        ind: np.ndarray,
        x_ind: np.ndarray,
        y_ind: np.ndarray,
    ) -> np.ndarray | None:
        """Reshapes True observation values to windows shape."""
        window_size = tl.get_obs_window_size(x_ind, y_ind)
        if window_size is not None:
            return self._obs_data[ind].reshape(window_size)
        return None

    @staticmethod
    def _regrid_cf(storage: dict, i: int, j: int, data: np.ndarray) -> dict:
        """Calculates average cloud fraction value to grid point."""
        data_ma = ma.array(data) if not isinstance(data, ma.MaskedArray) else data
        for key, downsample in storage.items():
            downsample[i, j] = ma.mean(data_ma)
            if "_A" in key and not data_ma.mask.all():
                downsample[i, j] = tl.average_column_sum(data_ma)
            storage[key] = downsample
        return storage

    def _regrid_iwc(
        self,
        storage: dict,
        i: int,
        j: int,
        ind_rain: np.ndarray,
        ind_no_rain: np.ndarray,
    ) -> dict:
        """Calculates average iwc value for each grid point."""
        for key, down_sample in storage.items():
            down_sample[i, j] = ma.masked
            if "rain" not in key:
                no_rain_data = self._obs_data[ind_no_rain]
                if ind_no_rain.any() and not no_rain_data.mask.all():
                    down_sample[i, j] = ma.mean(no_rain_data)
            if "rain" in key:
                rain_data = self._obs_data[ind_rain]
                if ind_rain.any() and not rain_data.mask.all():
                    down_sample[i, j] = ma.mean(rain_data)
            if "att" in key:
                no_rain_att_data = self._obs_obj.data["iwc_att"][ind_no_rain]
                if ind_no_rain.any() and not no_rain_att_data.mask.all():
                    down_sample[i, j] = ma.mean(no_rain_att_data)
            storage[key] = down_sample
        return storage

    def _regrid_product(self, storage: dict, i: int, j: int, ind: np.ndarray) -> dict:
        """Calculates average of standard product value for each grid point."""
        for key, down_sample in storage.items():
            obs_data_selected = ma.masked_invalid(self._obs_data[ind])
            down_sample[i, j] = (
                ma.mean(obs_data_selected)
                if not obs_data_selected.mask.all()
                else ma.masked
            )
            storage[key] = down_sample
        return storage

    def _append_data2object(self, data_storage: list) -> None:
        for storage in data_storage:
            for key in storage:
                down_sample = storage[key]
                self.model_obj.append_data(
                    down_sample,
                    f"{key}_{self.model_obj.model}{self.model_obj.cycle}",
                )
