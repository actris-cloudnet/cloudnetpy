from typing import Tuple, Union

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
            int(model_obj.resolution_h), ma.array(model_obj.wind), 1
        )
        time_steps = utils.binvec(self._model_time)
        self._time_steps = tl.time2datetime(time_steps, self._date)
        self._generate_downsample_product()

    def _generate_downsample_product(self):
        """Downsampling products are generated with different averaging methods
        for a selected size of model time-height window.
        """
        product_dict, product_adv_dict = self._get_method_storage()
        model_t = tl.time2datetime(self._model_time, self._date)
        for i in range(len(self._time_steps) - 1):
            x_ind = tl.get_1d_indices(
                (self._time_steps[i], self._time_steps[i + 1]), self._obs_time
            )
            if self._obs_obj.obs == "iwc":
                x_ind_no_rain = tl.get_1d_indices(
                    (self._time_steps[i], self._time_steps[i + 1]),
                    self._obs_time,
                    mask=self._obs_obj.data["iwc_rain"][:],
                )
            y_steps = tl.rebin_edges(self._model_height[i])
            for j in range(len(y_steps) - 1):
                x_ind_adv = tl.get_adv_indices(model_t[i], self._time_adv[i, j], self._obs_time)
                y_ind = tl.get_1d_indices((y_steps[j], y_steps[j + 1]), self._obs_height)
                ind = np.outer(x_ind, y_ind)
                ind_avd = np.outer(x_ind_adv, y_ind)
                if self._obs_obj.obs == "cf":
                    data = self._reshape_data_to_window(ind, x_ind, y_ind)
                    if data is None:
                        continue
                    product_dict = self._regrid_cf(product_dict, i, j, data)
                    data_adv = self._reshape_data_to_window(ind_avd, x_ind_adv, y_ind)
                    assert data_adv is not None
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
                    product_dict = self._regrid_iwc(product_dict, i, j, ind, ind_no_rain)
                    product_adv_dict = self._regrid_iwc(
                        product_adv_dict, i, j, ind_avd, ind_no_rain_adv
                    )
                else:
                    product_dict = self._regrid_product(product_dict, i, j, ind)
                    product_adv_dict = self._regrid_product(product_adv_dict, i, j, ind_avd)
        self._append_data2object([product_dict, product_adv_dict])

    def _get_method_storage(self):
        if self._obs_obj.obs == "cf":
            return self._cf_method_storage()
        if self._obs_obj.obs == "iwc":
            return self._iwc_method_storage()
        return self._product_method_storage()

    def _cf_method_storage(self) -> Tuple[dict, dict]:
        cf_dict = {
            "cf_V": np.zeros(self._model_height.shape),
            "cf_A": np.zeros(self._model_height.shape),
        }
        cf_adv_dict = {
            "cf_V_adv": np.zeros(self._model_height.shape),
            "cf_A_adv": np.zeros(self._model_height.shape),
        }
        return cf_dict, cf_adv_dict

    def _iwc_method_storage(self) -> Tuple[dict, dict]:
        iwc_dict = {
            "iwc": np.zeros(self._model_height.shape),
            "iwc_att": np.zeros(self._model_height.shape),
            "iwc_rain": np.zeros(self._model_height.shape),
        }
        iwc_adv_dict = {
            "iwc_adv": np.zeros(self._model_height.shape),
            "iwc_att_adv": np.zeros(self._model_height.shape),
            "iwc_rain_adv": np.zeros(self._model_height.shape),
        }
        return iwc_dict, iwc_adv_dict

    def _product_method_storage(self) -> Tuple[dict, dict]:
        product_dict = {f"{self._obs_obj.obs}": np.zeros(self._model_height.shape)}
        product_adv_dict = {f"{self._obs_obj.obs}_adv": np.zeros(self._model_height.shape)}
        return product_dict, product_adv_dict

    @staticmethod
    def _regrid_cf(storage: dict, i: int, j: int, data: Union[np.ndarray, None]) -> dict:
        """Calculates average cloud fraction value to grid point"""
        for key, downsample in storage.items():
            if data is not None:
                downsample[i, j] = np.nanmean(data)
                if "_A" in key:
                    downsample[i, j] = tl.average_column_sum(data)
            else:
                downsample[i, j] = np.nan
            storage[key] = downsample
        return storage

    def _reshape_data_to_window(
        self, ind: np.ndarray, x_ind: np.ndarray, y_ind: np.ndarray
    ) -> Union[None, np.ndarray]:
        """Reshapes True observation values to windows shape"""
        window_size = tl.get_obs_window_size(x_ind, y_ind)
        if window_size is not None:
            return self._obs_data[ind].reshape(window_size)
        return None

    def _regrid_iwc(
        self, storage: dict, i: int, j: int, ind_rain: np.ndarray, ind_no_rain: np.ndarray
    ) -> dict:
        """Calculates average iwc value for grid point"""
        for key, downsample in storage.items():
            if not self._obs_data[ind_no_rain].mask.all():
                downsample[i, j] = np.nanmean(self._obs_data[ind_no_rain])
            elif "rain" in key and not self._obs_data[ind_rain].mask.all():
                downsample[i, j] = np.nanmean(self._obs_data[ind_rain])
            else:
                downsample[i, j] = np.nan
            if "att" in key:
                iwc_att = self._obs_obj.data["iwc_att"][:]
                if iwc_att[ind_no_rain].mask.all():
                    downsample[i, j] = np.nan
                else:
                    downsample[i, j] = np.nanmean(iwc_att[ind_no_rain])
            storage[key] = downsample
        return storage

    def _regrid_product(self, storage: dict, i: int, j: int, ind: np.ndarray) -> dict:
        """Calculates average of standard product value to grid point"""
        for key, down_sample in storage.items():
            if not self._obs_data[ind].mask.all() and ind.any():
                down_sample[i, j] = np.nanmean(self._obs_data[ind])
            else:
                down_sample[i, j] = np.nan
            storage[key] = down_sample
        return storage

    def _append_data2object(self, data_storage: list):
        for storage in data_storage:
            for key in storage.keys():
                down_sample = storage[key]
                self.model_obj.append_data(
                    down_sample, f"{key}_{self.model_obj.model}{self.model_obj.cycle}"
                )
