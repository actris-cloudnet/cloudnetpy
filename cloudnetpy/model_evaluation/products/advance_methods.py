import importlib
import logging
from typing import Tuple

import numpy as np
from numpy import ma
from scipy.special import gamma

import cloudnetpy.utils as cl_tools
from cloudnetpy.datasource import DataSource
from cloudnetpy.model_evaluation.products.model_products import ModelManager
from cloudnetpy.model_evaluation.products.observation_products import ObservationManager


class AdvanceProductMethods(DataSource):
    """Class that adds advance methods of product to nc-file.
    Different methods could be filtering or adding info by making
    assumptions of model or observation data.

    Args:
        model_obj (object): The :class:'ModelManager' object.
        obs_obj (object): The :class:'ObservationManager' object.
    """

    def __init__(self, model_obj: ModelManager, model_file: str, obs_obj: ObservationManager):
        super().__init__(model_file)
        self._obs_obj = obs_obj
        self.product = obs_obj.obs
        self._date = obs_obj.date
        self._obs_height = obs_obj.data["height"][:]
        self._obs_data = obs_obj.data[obs_obj.obs][:]
        self._model_obj = model_obj
        self._model_time = model_obj.time
        self._model_height = model_obj.data[model_obj.keys["height"]][:]
        self.generate_products()

    def generate_products(self):
        cls = getattr(importlib.import_module(__name__), "AdvanceProductMethods")
        try:
            name = f"get_advance_{self.product}"
            getattr(cls, name)(self)
        except AttributeError as error:
            logging.warning(f"No advance method for {self.product}: {error}")

    def get_advance_cf(self):
        self.cf_cirrus_filter()

    def cf_cirrus_filter(self):
        cf = self.getvar_from_object("cf")
        h = self.getvar_from_object("h")
        temperature = self.getvar("temperature")
        t_screened = self.remove_extra_levels(temperature - 273.15)
        iwc, lwc = [self._model_obj.get_water_continent(var) for var in ["iwc", "lwc"]]
        tZT, tT, tZ, t = self.set_frequency_parameters()
        z_sen = self.fit_z_sensitivity(h)
        cf_filtered = self.filter_high_iwc_low_cf(cf, iwc, lwc)
        cloud_iwc, ice_ind = self.find_ice_in_clouds(cf_filtered, iwc, lwc)
        variance_iwc = self.iwc_variance(h, ice_ind)
        # Looks suspicious, check me:
        for i, ind in enumerate(zip(ice_ind[0], ice_ind[-1])):
            iwc_dist = self.calculate_iwc_distribution(cloud_iwc[i], variance_iwc[i])
            p_iwc = self.gamma_distribution(iwc_dist, variance_iwc[i], cloud_iwc[i])
            if np.sum(p_iwc) == 0 or p_iwc[-1] > 0.01 * np.sum(p_iwc):
                cf_filtered[ind] = np.nan
                continue
            obs_index = self.get_observation_index(
                iwc_dist, tZT, tT, tZ, t, float(t_screened[ind]), float(z_sen[ind])
            )
            cf_filtered[ind] = self.filter_cirrus(p_iwc, obs_index, cf_filtered[ind])
        cf_filtered[cf_filtered < 0.05] = ma.masked
        self._model_obj.append_data(
            cf_filtered, f"{self._model_obj.model}{self._model_obj.cycle}_cf_cirrus"
        )

    def getvar_from_object(self, arg: str) -> np.ndarray:
        v_name = arg if arg == "cf" else self._model_obj.get_model_var_names((arg,))[0]
        key = f"{self._model_obj.model}{self._model_obj.cycle}_{v_name}"
        return self._model_obj.data[key][:]

    def remove_extra_levels(self, arg: np.ndarray) -> np.ndarray:
        return self._model_obj.cut_off_extra_levels(arg)

    def set_frequency_parameters(self) -> Tuple:
        assert self._obs_obj.radar_freq is not None
        if 30 <= self._obs_obj.radar_freq <= 40:
            return 0.000242, -0.0186, 0.0699, -1.63
        if 90 <= float(self._obs_obj.radar_freq) <= 100:
            return 0.00058, -0.00706, 0.0923, -0.992
        raise ValueError

    def fit_z_sensitivity(self, h: np.ndarray) -> np.ndarray:
        assert self._obs_obj.z_sensitivity is not None
        assert self._obs_obj.height is not None
        z_sen = [
            cl_tools.rebin_1d(self._obs_obj.height, self._obs_obj.z_sensitivity, h[i])
            for i in range(len(h))
        ]
        return np.asarray(z_sen)

    def filter_high_iwc_low_cf(
        self, cf: np.ndarray, iwc: np.ndarray, lwc: np.ndarray
    ) -> np.ndarray:
        cf_filtered = self.mask_weird_indices(cf, iwc, lwc)
        if np.sum((iwc > 0) & (lwc < iwc / 10) & (cf_filtered > 0)) == 0:
            raise ValueError("No ice clouds in a input data")
        return cf_filtered

    @staticmethod
    def mask_weird_indices(cf: np.ndarray, iwc: np.ndarray, lwc: np.ndarray) -> np.ndarray:
        cf_filtered = np.copy(cf)
        weird_ind = (iwc / cf > 0.5e-3) & (cf < 0.001)
        weird_ind = weird_ind | (iwc == 0) & (lwc == 0) & (cf == 0)
        cf_filtered[weird_ind] = ma.masked
        return cf_filtered

    def find_ice_in_clouds(
        self, cf_filtered: np.ndarray, iwc: np.ndarray, lwc: np.ndarray
    ) -> Tuple[np.ndarray, tuple]:
        ice_ind = self.get_ice_indices(cf_filtered, iwc, lwc)
        cloud_iwc = iwc[ice_ind] / cf_filtered[ice_ind] * 1e3
        return cloud_iwc, ice_ind

    @staticmethod
    def get_ice_indices(cf_filtered: np.ndarray, iwc: np.ndarray, lwc: np.ndarray) -> tuple:
        return tuple(np.where((cf_filtered > 0) & (iwc > 0) & (lwc < iwc / 10)))

    def iwc_variance(self, height: np.ndarray, ice_ind: tuple) -> np.ndarray:
        u = self.getvar("uwind")
        v = self.getvar("vwind")
        u = self.remove_extra_levels(u)
        v = self.remove_extra_levels(v)
        w_shear = self.calculate_wind_shear(self._model_obj.wind, u, v, height)
        variance_iwc = self.calculate_variance_iwc(w_shear, ice_ind)
        return variance_iwc

    def calculate_variance_iwc(self, w_shear: np.ndarray, ice_ind: tuple) -> np.ndarray:
        return 10 ** (0.3 * np.log10(self._model_obj.resolution_h) - 0.04 * w_shear[ice_ind] - 1.03)

    @staticmethod
    def calculate_wind_shear(wind, u: np.ndarray, v: np.ndarray, height: np.ndarray) -> np.ndarray:
        grand_winds = []
        for w in (wind, u, v):
            grad_w = np.zeros(w.shape)
            grad_w[0, :] = (w[1, :] - w[0, :]) / (height[1, :] - height[0, :])
            grad_w[1:-2, :] = (w[2:-1, :] - 2 * w[1:-2, :] + w[1:-2, :]) / (
                height[2:-1, :] - height[1:-2, :]
            )
            grad_w[-1, :] = (w[-1, :] - w[-2, :]) / (height[-1, :] - height[-2, :])
            grand_winds.append(grad_w)

        w_shear = np.sqrt(np.power(grand_winds[1], 2) + np.power(grand_winds[-1], 2))
        w_shear[grand_winds[0] < 0] = 0 - w_shear[grand_winds[0] < 0]
        return w_shear

    @staticmethod
    def calculate_iwc_distribution(
        cloud_iwc: float, f_variance_iwc: float, n_std: int = 5, n_dist: int = 250
    ) -> np.ndarray:
        finish = cloud_iwc + n_std * (np.sqrt(f_variance_iwc) * cloud_iwc)
        iwc_dist = np.arange(0, finish, finish / (n_dist - 1))
        if cloud_iwc < iwc_dist[2]:
            finish = cloud_iwc * 10
            iwc_dist = np.arange(0, finish, finish / n_dist - 1)
        return iwc_dist

    @staticmethod
    def gamma_distribution(
        iwc_dist: np.ndarray, f_variance_iwc: float, cloud_iwc: float
    ) -> np.ndarray:
        def calculate_gamma_dist():
            alpha = 1 / f_variance_iwc
            return (
                1
                / gamma(alpha)
                * (alpha / cloud_iwc) ** alpha
                * iwc_dist[i] ** (alpha - 1)
                * ma.exp(-(alpha * iwc_dist[i] / cloud_iwc))
            )

        p_iwc = np.zeros(iwc_dist.shape)
        for i in range(len(iwc_dist)):
            p_iwc[i] = calculate_gamma_dist()
        return p_iwc

    @staticmethod
    def get_observation_index(
        iwc_dist: np.ndarray,
        tZT: float,
        tT: float,
        tZ: float,
        t: np.ndarray,
        temperature: float,
        z_sen: float,
    ) -> np.ndarray:
        def calculate_min_iwc():
            min_iwc = 10 ** (tZT * z_sen * temperature + tT * temperature + tZ * z_sen + t)
            return min_iwc

        iwc_min = calculate_min_iwc()
        obs_index = iwc_dist > iwc_min
        return obs_index

    @staticmethod
    def filter_cirrus(
        p_iwc: np.ndarray, obs_index: np.ndarray, cf_filtered: np.ndarray
    ) -> np.ndarray:
        return (np.sum(p_iwc * obs_index) / np.sum(p_iwc)) * cf_filtered
