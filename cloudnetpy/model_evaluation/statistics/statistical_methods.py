import logging
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from numpy import ma


class DayStatistics:
    """Class for calculating statistical analysis of day scale products.

    Class generates one statistical method at the time with given model data
    and observation data of wanted product.

    Args:
        method (str): Name on statistical method to be calculated
        product_info (list): List of information of statistical analysis is
            done with. A list includes observed product name (str), model variable (str)
            name and a name of observation variable (str). Example: ['cf', 'ECMWF',
            'Cloud fraction by volume']
        model (npt.NDArray): Ndarray of model simulation of product
        observation (npt.NDArray): Ndrray of Downsampled observation of product

    Raises:
        RuntimeError: A function of given method not found

    Returns:
        day_statistic (object): The :class:'DayStatistic' object.

    Examples:
        >>> from cloudnetpy.model_evaluation.products.product_resampling import \
        process_L3_day_product
        >>> method = 'error'
        >>> product_info = ['cf', 'European Centre for Medium-Range Weather Forecasts',
        'ecmwf']
        >>> model_array = np.array([[1,1,1],[1,1,1],[1,1,1]])
        >>> obs_array = np.array([[1,1,1],[1,1,1],[1,1,1]])
        >>> day_stat = DayStatistics(method, product_info, model_array, obs_array)
    """

    def __init__(
        self,
        method: str,
        product_info: list,
        model: npt.NDArray,
        observation: npt.NDArray,
    ) -> None:
        self.method = method
        self.product = product_info
        self.model_data = model
        self.observation_data = observation
        self._generate_day_statistics()

    def _get_method_attr(self) -> tuple[Callable, tuple]:
        params = (self.model_data, self.observation_data)
        if self.method == "error":
            return relative_error, params
        if self.method == "aerror":
            return absolute_error, params
        if self.method == "area":
            return calc_common_area_sum, params
        if self.method == "hist":
            return histogram, (self.product, *params)
        if self.method == "vertical":
            return vertical_profile, params
        msg = f"Unknown statistical method: {self.method}"
        raise RuntimeError(msg)

    def _generate_day_statistics(self) -> None:
        method, params = self._get_method_attr()
        try:
            self.model_stat, self.observation_stat = method(*params)
            self.title = day_stat_title(self.method, self.product)
        except RuntimeError:
            msg = f"Failed to calculate {self.method} of {self.product[0]}"
            logging.exception(msg)


def relative_error(
    model: ma.MaskedArray,
    observation: ma.MaskedArray,
) -> tuple[float, str]:
    model, observation = combine_masked_indices(model, observation)
    # Very small observation values make the relative error overflow; the
    # resulting huge values are expected, so ignore the warning.
    with np.errstate(divide="ignore", over="ignore"):
        error = ((model - observation) / observation) * 100
    return np.round(error, 2), ""


def absolute_error(
    model: ma.MaskedArray,
    observation: ma.MaskedArray,
) -> tuple[float, str]:
    model, observation = combine_masked_indices(model, observation)
    error = (observation - model) * 100
    return np.round(error, 2), ""


def combine_masked_indices(
    model: ma.MaskedArray,
    observation: ma.MaskedArray,
) -> tuple[ma.MaskedArray, ma.MaskedArray]:
    """Connects two array masked indices to one and add in two array same mask."""
    observation[np.where(np.isnan(observation))] = ma.masked
    model[model < np.min(observation)] = ma.masked
    combine_mask = model.mask + observation.mask
    model[combine_mask] = ma.masked
    observation[combine_mask] = ma.masked
    return model, observation


def calc_common_area_sum(
    model: ma.MaskedArray,
    observation: ma.MaskedArray,
) -> tuple[float, str]:
    def _indices_of_mask_sum() -> float:
        # Calculate percentage value of common area of indices from two arrays.
        # Results is total number of common indices with value
        observation[np.where(np.isnan(observation))] = ma.masked
        model[np.where(np.isnan(model))] = ma.masked
        model[model < np.min(observation)] = ma.masked
        combine_mask = model.mask + observation.mask
        common_mask = np.bitwise_and(model.mask, observation.mask)
        valid_area = np.sum(~common_mask)
        if valid_area == 0:
            return 0.0
        the_match = np.sum(~combine_mask) / valid_area * 100
        return np.round(the_match, 2)

    match = _indices_of_mask_sum()
    return match, ""


def histogram(
    product: list,
    model: ma.MaskedArray,
    observation: ma.MaskedArray,
) -> tuple:
    if "cf" in product:
        model = ma.round(model[~model.mask].data, decimals=1).flatten()
        observation = ma.round(
            observation[~observation.mask].data,
            decimals=1,
        ).flatten()
    else:
        model = ma.round(model[~model.mask].data, decimals=6).flatten()
        observation = ma.round(
            observation[~observation.mask].data,
            decimals=6,
        ).flatten()
    observation = observation[~np.isnan(observation)]
    hist_bins = np.histogram(observation, density=True)[-1]
    model[model > hist_bins[-1]] = hist_bins[-1]
    return model, observation


def vertical_profile(model: ma.MaskedArray, observation: ma.MaskedArray) -> tuple:
    # L3 fields are shaped (time, level); average over time (axis 0) to get the
    # vertical profile over height.
    model_vertical = ma.mean(model, axis=0)
    obs_vertical = np.nanmean(observation, axis=0)
    return model_vertical, obs_vertical


def day_stat_title(method: str, product: list) -> str | tuple[str, str]:
    if method in ("hist", "vertical"):
        return f"{product[1]}", f"{product[-1]}"
    return f"{product[-1]} vs {product[1]}"
