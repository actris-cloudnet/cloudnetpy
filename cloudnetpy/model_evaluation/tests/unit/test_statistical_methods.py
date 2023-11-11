import numpy as np
import pytest
from numpy import ma, testing

import cloudnetpy.model_evaluation.statistics.statistical_methods as sts

PRODUCT_cf = ["cf", "ECMWF", "Cloud fraction"]
PRODUCT_iwc = ["iwc", "ECMWF", "Ice water content"]


def test_relative_error() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    observation = ma.array([[3, 2, 5, 4], [4, 6, 8, 4]])
    x, _ = sts.relative_error(model, observation)
    compare = ma.array([[-66.67, 0.0, -60.0, -25.0], [-50.0, -33.33, 25.0, -75.0]])
    testing.assert_array_almost_equal(x, compare)


def test_relative_error_mask() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 1, 10, 1]])
    model.mask = np.array([[0, 0, 0, 1], [0, 1, 0, 1]])
    observation = ma.array([[1, 2, 5, 4], [4, 6, 8, 1]])
    observation.mask = np.array([[1, 0, 0, 1], [0, 0, 0, 1]])
    x, _ = sts.relative_error(model, observation)
    compare = ma.array(
        [
            ma.array([-99, 0.0, -60.0, -99], mask=[1, 0, 0, 1]),
            ma.array([-50.0, -83.33, 25.0, -99], mask=[0, 0, 0, 1]),
        ],
    )
    testing.assert_array_almost_equal(x, compare)


def test_relative_error_nan() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    observation = ma.array([[3, 2, 5, np.nan], [4, np.nan, 8, 4]])
    x, _ = sts.relative_error(model, observation)
    compare = ma.array(
        [
            ma.array([-66.67, 0.0, -60.0, -99], mask=[0, 0, 0, 1]),
            ma.array([-50.0, -99, 25.0, -75.0], mask=[0, 1, 0, 0]),
        ],
    )
    testing.assert_array_almost_equal(x, compare)


def test_absolute_error() -> None:
    model = ma.array([[0.1, 0.2, 0.2, 0.3], [0.2, 0.4, 1.0, 0.0]])
    observation = ma.array([[0.2, 0.2, 0.1, 0.4], [0.4, 0.6, 0.8, 0.2]])
    x, _ = sts.absolute_error(model, observation)
    compare = ma.array([[10.0, 0.0, -10.0, 10.0], [20.0, 20.0, -20.0, 20.0]])
    testing.assert_array_almost_equal(x, compare)


def test_absolute_error_nan() -> None:
    model = ma.array([[0.1, 0.2, 0.2, 0.3], [0.2, 0.4, 1.0, 0.0]])
    observation = ma.array([[0.2, np.nan, 0.1, 0.4], [np.nan, 0.6, 0.8, 0.2]])
    x, _ = sts.absolute_error(model, observation)
    compare = ma.array(
        [
            ma.array([10.0, -99, -10.0, 10.0], mask=[0, 1, 0, 0]),
            ma.array([-99, 20.0, -20.0, 20.0], mask=[1, 0, 0, 0]),
        ],
    )
    testing.assert_array_almost_equal(x, compare)


def test_absolute_error_mask() -> None:
    model = ma.array([[0.1, 0.2, 0.2, 0.3], [0.2, 0.4, 1.0, 0.0]])
    model.mask = np.array([[0, 0, 0, 1], [0, 1, 0, 1]])
    observation = ma.array([[0.2, 0.2, 0.1, 0.4], [0.4, 0.6, 0.8, 0.2]])
    observation.mask = np.array([[0, 0, 0, 0], [0, 1, 0, 0]])
    x, _ = sts.absolute_error(model, observation)
    compare = ma.array(
        [
            ma.array([10.0, 0.0, -10.0, -99], mask=[0, 0, 0, 1]),
            ma.array([20.0, -99, -20.0, -99], mask=[0, 1, 0, 1]),
        ],
    )
    testing.assert_array_almost_equal(x, compare)


def test_combine_masked_indices() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    observation = ma.array([[3, 2, 5, 4], [4, 6, 8, 4]])
    x, y = sts.combine_masked_indices(model, observation)
    compare_m = ma.array([[4, 2, 2, 3], [2, 4, 10, 6]])
    compare_o = ma.array([[3, 2, 5, 4], [4, 6, 8, 4]])
    testing.assert_array_almost_equal(x, compare_m)
    testing.assert_array_almost_equal(y, compare_o)


def test_combine_masked_indices_min() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    observation = ma.array([[3, 2, 5, 4], [4, 6, 8, 4]])
    x, y = sts.combine_masked_indices(model, observation)

    compare_m = ma.array(
        [
            ma.array([-99, 2, 2, 3], mask=[1, 0, 0, 0]),
            ma.array([2, 4, 10, -99], mask=[1, 0, 0, 1]),
        ],
    )
    compare_o = ma.array(
        [
            ma.array([-99, 2, 5, 4], mask=[1, 0, 0, 0]),
            ma.array([4, 6, 8, -99], mask=[0, 0, 0, 1]),
        ],
    )

    testing.assert_array_almost_equal(x, compare_m)
    testing.assert_array_almost_equal(y, compare_o)


def test_combine_masked_indices_mask() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    model.mask = ma.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    observation = ma.array([[3, 2, 1, 4], [4, 6, 8, 4]])
    observation.mask = ma.array([[0, 1, 1, 0], [1, 0, 0, 0]])
    x, y = sts.combine_masked_indices(model, observation)
    model = ma.array(
        [
            ma.array([-99, -99, -99, 3], mask=[1, 1, 1, 0]),
            ma.array([-99, 4, 10, -99], mask=[1, 0, 0, 1]),
        ],
    )
    observation = ma.array(
        [
            ma.array([-99, -99, -99, 4], mask=[1, 1, 1, 0]),
            ma.array([-99, 6, 8, -99], mask=[1, 0, 0, 1]),
        ],
    )
    testing.assert_array_almost_equal(x, model)
    testing.assert_array_almost_equal(y, observation)


def test_combine_masked_indices_nan() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    observation = ma.array([[np.nan, 2, 5, 4], [4, 6, np.nan, 4]])
    x, y = sts.combine_masked_indices(model, observation)
    model = ma.array(
        [
            ma.array([-99, 2, 2, 3], mask=[1, 0, 0, 0]),
            ma.array([2, 4, -99, 1], mask=[0, 0, 1, 0]),
        ],
    )
    testing.assert_array_almost_equal(x, model)
    testing.assert_array_almost_equal(y, observation)


def test_calc_common_area_sum() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    observation = ma.array([[3, 2, 1, 4], [4, 6, 8, 4]])
    x, _ = sts.calc_common_area_sum(model, observation)
    testing.assert_almost_equal(x, 100)


def test_calc_common_area_sum_min() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    model.mask = ma.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    observation = ma.array([[3, 2, 1, 4], [4, 6, 8, 4]])
    observation.mask = ma.array([[0, 1, 1, 0], [1, 0, 0, 0]])
    x, _ = sts.calc_common_area_sum(model, observation)
    testing.assert_almost_equal(x, 60.0)


def test_calc_common_area_sum_nan() -> None:
    model = ma.array([[1, 2, 2, 3], [2, np.nan, 10, 1]])
    model.mask = ma.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    observation = ma.array([[3, 2, 1, 4], [4, 6, np.nan, 4]])
    observation.mask = ma.array([[0, 0, 1, 0], [1, 0, 0, 0]])
    x, _ = sts.calc_common_area_sum(model, observation)
    testing.assert_almost_equal(x, 25.0)


def test_calc_common_area_sum_mask() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    model.mask = ma.array([[1, 0, 0, 0], [0, 0, 0, 0]])
    observation = ma.array([[3, 2, 1, 4], [4, 6, 8, 4]])
    observation.mask = ma.array([[0, 0, 1, 0], [1, 0, 0, 0]])
    x, _ = sts.calc_common_area_sum(model, observation)
    testing.assert_almost_equal(x, 50.0)


def test_histogram() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    observation = ma.array([[3, 2, 1, 4], [4, 6, 8, 4]])
    compare_x = np.array([1, 2, 2, 3, 2, 4, 8, 1])
    compare_y = np.array([3, 2, 1, 4, 4, 6, 8, 4])
    x, y = sts.histogram(PRODUCT_iwc, model, observation)
    testing.assert_array_almost_equal(x, compare_x)
    testing.assert_array_almost_equal(y, compare_y)


def test_histogram_mask() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    model.mask = ma.array([[1, 0, 0, 0], [0, 0, 0, 1]])
    observation = ma.array([[3, 2, 1, 4], [4, 6, 8, 4]])
    observation.mask = ma.array([[0, 1, 1, 0], [1, 0, 0, 0]])
    compare_x = ma.array([2, 2, 3, 2, 4, 8])
    compare_y = ma.array([3, 4, 6, 8, 4])
    x, y = sts.histogram(PRODUCT_iwc, model, observation)
    testing.assert_array_almost_equal(x, compare_x)
    testing.assert_array_almost_equal(y, compare_y)


def test_histogram_nan() -> None:
    model = ma.array([[1, 2, 2, 3], [2, 4, 10, 1]])
    observation = ma.array([[3, np.nan, 1, 4], [np.nan, 6, np.nan, 4]])
    compare_x = np.array([1, 2, 2, 3, 2, 4, 6, 1])
    compare_y = np.array([3, 1, 4, 6, 4])
    x, y = sts.histogram(PRODUCT_iwc, model, observation)
    testing.assert_array_almost_equal(x, compare_x)
    testing.assert_array_almost_equal(y, compare_y)


def test_vertical_profile() -> None:
    model = ma.array([[0, 2, 2, 3], [2, 4, 10, 1], [4, 6, 6, 8]])
    observation = ma.array([[3, 1, 1, 4], [4, 3, 8, 0], [5, 5, 3, 2]])
    x, y = sts.vertical_profile(model, observation)
    model = ma.array([2, 4, 6, 4])
    observation = ma.array([4, 3, 4, 2])
    testing.assert_array_almost_equal(x, model)
    testing.assert_array_almost_equal(y, observation)


def test_vertical_profile_nan() -> None:
    model = ma.array([[0, 2, 2, 3], [2, 4, 10, 1], [4, 6, 6, 8]])
    observation = ma.array([[3, 1, 1, np.nan], [np.nan, 3, 8, 0], [5, 5, 3, 2]])
    x, y = sts.vertical_profile(model, observation)
    model = ma.array([2, 4, 6, 4])
    observation = ma.array([4, 3, 4, 1])
    testing.assert_array_almost_equal(x, model)
    testing.assert_array_almost_equal(y, observation)


def test_vertical_profile_mask() -> None:
    model = ma.array([[0, 2, 2, 3], [2, 4, 10, 1], [4, 6, 6, 8]])
    model.mask = ma.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    observation = ma.array([[3, 1, 1, 4], [4, 3, 8, 0], [5, 5, 3, 2]])
    observation.mask = ma.array([[0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 0]])
    x, y = sts.vertical_profile(model, observation)
    model = ma.array([3, 4, 6, 5.5])
    observation = ma.array([4, -99, 4, 3], mask=[0, 1, 0, 0])
    testing.assert_array_almost_equal(x, model)
    testing.assert_array_almost_equal(y, observation)


@pytest.mark.parametrize(
    "method, title",
    [
        ("error", "Cloud fraction vs ECMWF"),
        ("vertical", ("ECMWF", "Cloud fraction")),
    ],
)
def test_day_stat_title(method, title) -> None:
    x = sts.day_stat_title(method, PRODUCT_cf)
    assert x == title
