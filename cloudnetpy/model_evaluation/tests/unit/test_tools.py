from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from numpy import ma, testing

from cloudnetpy.model_evaluation.products import tools
from cloudnetpy.model_evaluation.products.model_products import ModelManager

MODEL = "ecmwf"
PRODUCT = "iwc"


def test_time2datetime() -> None:
    time = np.array(range(10))
    date = datetime(2020, 4, 7, 0, 0, 0, tzinfo=timezone.utc)
    result = tools.time2datetime(time, date)
    expected = [
        datetime(2020, 4, 7, 0, 0, 0, tzinfo=timezone.utc) + timedelta(hours=1 * i)
        for i in range(10)
    ]
    assert all(a == b for a, b in zip(result, expected))


def test_rebin_edges() -> None:
    data = np.array([1, 3, 6, 10, 15, 21, 28])
    expected = np.array([-1, 2, 4.5, 8, 12.5, 18, 24.5, 35])
    result = tools.rebin_edges(data)
    testing.assert_array_almost_equal(result, expected)


def test_calculate_advection_time_hour(model_file) -> None:
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    resolution = model.resolution_h
    wind = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    sampling = 1
    expected = resolution * 1000 / wind / 60**2
    expected[expected > 1 / sampling] = 1 / sampling
    expected = np.asarray([[timedelta(hours=float(t)) for t in tt] for tt in expected])
    result = tools.calculate_advection_time(resolution, ma.array(wind), sampling)
    assert result.all() == expected.all()


def test_calculate_advection_time_10min(model_file) -> None:
    model = ModelManager(str(model_file), MODEL, PRODUCT)
    resolution = model.resolution_h
    wind = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    sampling = 6
    expected = resolution * 1000 / wind / 60**2
    expected[expected > 1 / sampling] = 1 / sampling
    expected = np.asarray([[timedelta(hours=float(t)) for t in tt] for tt in expected])
    result = tools.calculate_advection_time(resolution, ma.array(wind), sampling)
    assert result.all() == expected.all()


def test_calculate_advection_time_fractional_resolution() -> None:
    # A sub-kilometre / fractional resolution must not be truncated to int.
    resolution = 0.5
    wind = ma.array([[2.0]])
    sampling = 6
    result = tools.calculate_advection_time(resolution, wind, sampling)
    expected = timedelta(hours=resolution * 1000 / 2.0 / 60**2)
    assert result[0, 0] == expected


def test_get_1d_indices() -> None:
    window = (1, 5)
    data = ma.array([0, 1, 2, 3, 4, 5, 6, 7])
    expected = ma.array([0, 1, 1, 1, 1, 0, 0, 0])
    result = tools.get_1d_indices(window, data)
    testing.assert_array_almost_equal(result, expected)


def test_get_1d_indices_mask() -> None:
    window = (1, 5)
    data = ma.array([0, 1, 2, 3, 4, 5, 6, 7])
    mask = np.array([0, 0, 1, 0, 0, 1, 0, 1], dtype=bool)
    result = tools.get_1d_indices(window, data, mask)
    data[mask] = ma.masked
    expected = ma.array(
        [0, 1, -99, 1, 1, -99, 0, -99],
        mask=[False, False, True, False, False, True, False, True],
    )
    testing.assert_array_almost_equal(result, expected)


def test_get_adv_indices() -> None:
    model_t = 3
    adv_t = 4
    data = ma.array([0, 1, 2, 3, 4, 5, 6, 7])
    expected: ma.MaskedArray = ma.array([0, 1, 1, 1, 1, 0, 0, 0], dtype=bool)
    result = tools.get_adv_indices(model_t, adv_t, data)
    testing.assert_array_almost_equal(result, expected)


def test_get_adv_indices_mask() -> None:
    model_t = 3
    adv_t = 4
    data: ma.MaskedArray = ma.array([0, 1, 2, 3, 4, 5, 6, 7])
    mask: ma.MaskedArray = ma.array([0, 0, 1, 0, 0, 1, 0, 1], dtype=bool)
    result = tools.get_adv_indices(model_t, adv_t, data, mask)
    data[mask] = ma.masked
    expected = ma.array(
        [0, 1, -99, 1, 1, -99, 0, 0],
        mask=[False, False, True, False, False, True, False, False],
    )
    testing.assert_array_almost_equal(result, expected)


def test_obs_windows_size() -> None:
    ind_x = np.array([0, 0, 1, 1, 1, 1, 0], dtype=bool)
    ind_y = np.array([0, 1, 1, 1, 0, 0, 0], dtype=bool)
    result = tools.get_obs_window_size(ind_x, ind_y)
    assert result is not None
    testing.assert_almost_equal(result, (4, 3))


def test_obs_windows_size_none() -> None:
    ind_x = np.array([0, 0, 1, 1, 1, 1, 0], dtype=bool)
    ind_y = np.array([0, 0, 0, 0, 0, 0, 0], dtype=bool)
    result = tools.get_obs_window_size(ind_x, ind_y)
    assert result is None


def test_obs_windows_size_first_index() -> None:
    ind_x = np.array([1, 0, 0, 0], dtype=bool)
    ind_y = np.array([1, 0, 0, 0], dtype=bool)
    result = tools.get_obs_window_size(ind_x, ind_y)
    assert result is not None
    testing.assert_almost_equal(result, (1, 1))
