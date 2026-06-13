import numpy as np
from numpy import ma, testing

from cloudnetpy.model_evaluation.plotting import plot_tools as plt


def test_parse_wanted_names(regrid_file) -> None:
    expected = ["model_cf", "cf_V"]
    result, _ = plt.parse_wanted_names(regrid_file, "cf")
    assert result == expected


def test_parse_wanted_names_adv(regrid_file) -> None:
    expected = ["model_cf", "cf_V_adv"]
    _, result_adv = plt.parse_wanted_names(regrid_file, "cf")
    assert result_adv == expected


def test_parse_wanted_names_advance_False(regrid_file) -> None:
    expected = ["model_cf", "cf_V"]
    result, _ = plt.parse_wanted_names(regrid_file, "cf", advance=False)
    assert result == expected


def test_parse_wanted_names_advance_True(regrid_file) -> None:
    expected = ["model_cf", "model_cf_cirrus", "model_cf_snow", "cf_V"]
    result, _ = plt.parse_wanted_names(regrid_file, "cf", advance=True)
    assert result == expected


def test_parse_wanted_names_adv_advance_True(regrid_file) -> None:
    expected = ["model_cf", "model_cf_cirrus", "model_cf_snow", "cf_V_adv"]
    _, result_adv = plt.parse_wanted_names(regrid_file, "cf", advance=True)
    assert result_adv == expected


def test_parse_wanted_names_adv_advance_False(regrid_file) -> None:
    expected = ["model_cf", "cf_V_adv"]
    _, result_adv = plt.parse_wanted_names(regrid_file, "cf", advance=False)
    assert result_adv == expected


def test_parse_wanted_names_fixed_list(regrid_file) -> None:
    expected = ["model_cf", "model_cf_cirrus"]
    result, _ = plt.parse_wanted_names(regrid_file, "cf", variables=expected)
    assert result == expected


def test_parse_wanted_names_adv_fixed_list(regrid_file) -> None:
    expected = ["model_cf_cirrus", "cf_V_adv"]
    result_adv, _ = plt.parse_wanted_names(regrid_file, "cf", variables=expected)
    assert result_adv == expected


def test_sort_model2first_element() -> None:
    a = ["model_cf", "cf_V", "cf_A", "model_cf_cirrus"]
    expected = ["model_cf", "model_cf_cirrus", "cf_V", "cf_A"]
    result = plt.sort_model2first_element(a)
    assert result == expected


def test_read_data_characters(regrid_file) -> None:
    time = np.array([[2, 2], [6, 6], [10, 10]])
    height = np.array([[0.01, 0.014], [0.008, 0.014], [0.009, 0.015]])
    data = np.array([[0, 2], [3, 6], [5, 8]])
    result_data, result_time, result_height = plt.read_data_characters(
        regrid_file, "model_cf"
    )
    expected = [data, time, height]
    results = [result_data, result_time, result_height]
    for i in range(3):
        testing.assert_array_almost_equal(expected[i], results[i])


def test_mask_small_values_lwc() -> None:
    name = "lwc_lol"
    data = ma.array([[0, 1], [3, 6], [5, 8]])
    data = plt.mask_small_values(data, name)
    expected = ma.array([[0, 1], [3, 6], [5, 8]])
    expected[0, 0] = ma.masked
    testing.assert_array_almost_equal(data.mask, expected.mask)


def test_mask_small_values_lwc_mask() -> None:
    name = "lwc_lol"
    data = ma.array([[0, 0.000001], [3, 6], [5, 8]])
    data = plt.mask_small_values(data, name)
    expected = ma.array([[0, -0.000001], [3, 6], [5, 8]])
    expected[0, 0] = ma.masked
    expected[0, 1] = ma.masked
    testing.assert_array_almost_equal(data.mask, expected.mask)


def test_mask_small_values_iwc() -> None:
    name = "iwc_lol"
    data = ma.array([[0, 1], [3, 6], [5, 8]])
    data = plt.mask_small_values(data, name)
    expected = ma.array([[0, 1], [3, 6], [5, 8]])
    expected[0, 0] = ma.masked
    testing.assert_array_almost_equal(data.mask, expected.mask)


def test_mask_small_values_iwc_mask() -> None:
    name = "iwc_lol"
    data = ma.array([[0, 0.00000001], [3, 6], [5, 8]])
    data = plt.mask_small_values(data, name)
    expected = ma.array([[0, -0.000001], [3, 6], [5, 8]])
    expected[0, 0] = ma.masked
    expected[0, 1] = ma.masked
    testing.assert_array_almost_equal(data.mask, expected.mask)


def test_mask_small_values() -> None:
    name = "cf_lol"
    data = ma.array([[0, 1], [3, 6], [5, 8]])
    data = plt.mask_small_values(data, name)
    expected = ma.array([[0, 1], [3, 6], [5, 8]])
    expected[0, 0] = ma.masked
    testing.assert_array_almost_equal(data.mask, expected.mask)


def test_mask_small_values_mask() -> None:
    name = "cf_lol"
    data = ma.array([[0, -0.000001], [3, 6], [5, 8]])
    data_mask = plt.mask_small_values(data, name)
    expected = ma.array([[0, -0.000001], [3, 6], [5, 8]])
    expected[0, 0] = ma.masked
    expected[0, 1] = ma.masked
    testing.assert_array_equal(data_mask.mask, expected.mask)


def test_reshape_1d2nd() -> None:
    one_d = np.array([1, 2, 3, 4])
    two_d = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    expected = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    result = plt.reshape_1d2nd(one_d, two_d)
    testing.assert_array_almost_equal(result, expected)


def test_create_segment_values() -> None:
    model_mask = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=bool)
    model = ma.array(np.ones(model_mask.shape), mask=model_mask)
    obs_mask = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 1, 1]], dtype=bool)
    obs = ma.array(np.ones(obs_mask.shape), mask=obs_mask)
    result = plt.create_segment_values(model, obs)
    expected = np.array([[2, 3, 3, 2], [2, 2, 0, 2], [0, 2, 1, 1]])
    testing.assert_array_almost_equal(result, expected)


def test_rolling_mean() -> None:
    data = np.ma.array([1, 2, 7, 4, 2, 3, 8, 5])
    result = plt.rolling_mean(data, 2)
    expected = np.array([1.5, 4.5, 5.5, 3, 2.5, 5.5, 6.5, 5])
    testing.assert_array_almost_equal(result, expected)


def test_rolling_mean_nan() -> None:
    data = np.ma.array([1, 2, np.nan, 4, 2, np.nan, 8, 5])
    result = plt.rolling_mean(data, 2)
    expected = np.array([1.5, 2, 4, 3, 2, 8, 6.5, 5])
    testing.assert_array_almost_equal(result, expected)


def test_rolling_mean_mask() -> None:
    data = np.ma.array([1, 2, 7, 4, 2, 3, 8, 5])
    data.mask = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    result = plt.rolling_mean(data, 2)
    expected = np.array([1.5, 2, 4, 4, 3, 5.5, 8, np.nan])
    testing.assert_array_almost_equal(result, expected)


def test_rolling_mean_all_mask() -> None:
    data = np.ma.array([1, 2, 7, 4, 2, 3, 8, 5])
    data.mask = np.array([0, 1, 1, 1, 1, 0, 0, 1])
    result = plt.rolling_mean(data, 2)
    expected = np.array([1, np.nan, np.nan, np.nan, 3, 5.5, 8, np.nan])
    testing.assert_array_almost_equal(result, expected)


def test_change2one_dim_axes_maskY() -> None:
    axis_x = np.ma.array(
        [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]],
    )
    axis_y = np.ma.array(
        [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
    )
    axis_y[1] = np.ma.masked
    data = np.ma.array(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
    )
    result_x, result_y, result_data = plt.change2one_dim_axes(axis_x, axis_y, data)
    expected_x = np.array([1, 2, 3, 4])
    expected_y = np.array([1, 2, 3, 4, 5])
    testing.assert_array_almost_equal(result_x, expected_x)
    testing.assert_array_almost_equal(result_y, expected_y)


def test_change2one_dim_axes_maskX() -> None:
    axis_x = np.ma.array(
        [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]],
    )
    axis_y = np.ma.array(
        [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
    )
    axis_x[1] = np.ma.masked
    data = np.ma.array(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
    )
    result_x, result_y, result_data = plt.change2one_dim_axes(axis_x, axis_y, data)
    expected_x = np.array([1, 2, 3, 4])
    expected_y = np.array([1, 2, 3, 4, 5])
    testing.assert_array_almost_equal(result_x, expected_x)
    testing.assert_array_almost_equal(result_y, expected_y)


def test_change2one_dim_axes() -> None:
    axis_x = np.ma.array(
        [[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]],
    )
    expected_x = np.copy(axis_x)
    axis_y = np.ma.array(
        [[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]],
    )
    expected_y = np.copy(axis_y)
    data = np.ma.array(
        [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
    )
    result_x, result_y, result_data = plt.change2one_dim_axes(axis_x, axis_y, data)
    testing.assert_array_almost_equal(result_x, expected_x)
    testing.assert_array_almost_equal(result_y, expected_y)
