import numpy as np
from numpy import ma, testing

from cloudnetpy.model_evaluation.plotting import plot_tools as plt


def test_parse_wanted_names(regrid_file):
    """nc_file: str, name: str, model: str,
    vars: Union[list, None] = None,
    advance: bool = False"""
    compare = ["ecmwf_cf", "cf_ecmwf"]
    x, _ = plt.parse_wanted_names(regrid_file, "cf", "ecmwf")
    assert x == compare


def test_parse_wanted_names_adv(regrid_file):
    compare = ["ecmwf_cf", "cf_adv_ecmwf"]
    _, x_adv = plt.parse_wanted_names(regrid_file, "cf", "ecmwf")
    assert x_adv == compare


def test_parse_wanted_names_advance_False(regrid_file):
    compare = ["ecmwf_cf", "cf_ecmwf"]
    x, _ = plt.parse_wanted_names(regrid_file, "cf", "ecmwf", advance=False)
    assert x == compare


def test_parse_wanted_names_advance_True(regrid_file):
    compare = ["ecmwf_cf", "ecmwf_cf_cirrus", "ecmwf_cf_snow", "cf_ecmwf"]
    x, _ = plt.parse_wanted_names(regrid_file, "cf", "ecmwf", advance=True)
    assert x == compare


def test_parse_wanted_names_adv_advance_True(regrid_file):
    compare = ["ecmwf_cf", "ecmwf_cf_cirrus", "ecmwf_cf_snow", "cf_adv_ecmwf"]
    _, x_adv = plt.parse_wanted_names(regrid_file, "cf", "ecmwf", advance=True)
    assert x_adv == compare


def test_parse_wanted_names_adv_advance_False(regrid_file):
    compare = ["ecmwf_cf", "cf_adv_ecmwf"]
    _, x_adv = plt.parse_wanted_names(regrid_file, "cf", "ecmwf", advance=False)
    assert x_adv == compare


def test_parse_wanted_names_fixed_list(regrid_file):
    compare = ["ecmwf_cf", "ecmwf_cf_cirrus"]
    x, _ = plt.parse_wanted_names(regrid_file, "cf", "ecmwf", variables=compare)
    assert x == compare


def test_parse_wanted_names_adv_fixed_list(regrid_file):
    compare = ["ecmwf_cf_cirrus", "cf_adv_ecmwf"]
    x_adv, _ = plt.parse_wanted_names(regrid_file, "cf", "ecmwf", variables=compare)
    assert x_adv == compare


def test_sort_model2first_element():
    a = ["ec_i", "cf_ec_i", "cf_ec_ii", "ec_ii"]
    e = "ec"
    compare = ["ec_i", "ec_ii", "cf_ec_i", "cf_ec_ii"]
    x = plt.sort_model2first_element(a, e)
    assert x == compare


def test_sort_cycles_vars():
    a = ["era5_cf_1-12", "era5_cf_7-18", "cf_era5_1-12", "cf_era5_7-18"]
    e = "era5"
    compare = [["era5_cf_1-12", "cf_era5_1-12"], ["era5_cf_7-18", "cf_era5_7-18"]]
    x, _ = plt.sort_cycles(a, e)
    assert x == compare


def test_sort_cycles_simo():
    a = ["era5_cf_1-12", "era5_cf_7-18", "cf_era5_1-12", "cf_era5_7-18"]
    e = "era5"
    compare = ["1-12", "7-18"]
    _, y = plt.sort_cycles(a, e)
    assert y == compare


def test_sort_cycles_vars_missing():
    a = ["icon_cf_12-23", "icon_cf_36-47", "cf_icon_12-23", "cf_icon_36-47"]
    e = "icon"
    compare = [["icon_cf_12-23", "cf_icon_12-23"], ["icon_cf_36-47", "cf_icon_36-47"]]
    x, _ = plt.sort_cycles(a, e)
    assert x == compare


def test_sort_cycles_cycles_missing():
    a = ["icon_cf_12-23", "icon_cf_36-47", "cf_icon_12-23", "cf_icon_36-47"]
    e = "icon"
    compare = ["12-23", "36-47"]
    _, y = plt.sort_cycles(a, e)
    assert y == compare


def test_read_data_characters(regrid_file):
    t = np.array([[2, 2], [6, 6], [10, 10]])
    h = np.array([[0.01, 0.014], [0.008, 0.014], [0.009, 0.015]])
    data = np.array([[0, 2], [3, 6], [5, 8]])
    x, y, z = plt.read_data_characters(regrid_file, "ecmwf_cf", "ecmwf")
    compare = [data, t, h]
    test = [x, y, z]
    for i in range(3):
        testing.assert_array_almost_equal(compare[i], test[i])


def test_mask_small_values_lwc():
    name = "lwc_lol"
    data = ma.array([[0, 1], [3, 6], [5, 8]])
    data = plt.mask_small_values(data, name)
    compare = ma.array([[0, 1], [3, 6], [5, 8]])
    compare[0, 0] = ma.masked
    testing.assert_array_almost_equal(data.mask, compare.mask)


def test_mask_small_values_lwc_mask():
    name = "lwc_lol"
    data = ma.array([[0, 0.000001], [3, 6], [5, 8]])
    data = plt.mask_small_values(data, name)
    compare = ma.array([[0, -0.000001], [3, 6], [5, 8]])
    compare[0, 0] = ma.masked
    compare[0, 1] = ma.masked
    testing.assert_array_almost_equal(data.mask, compare.mask)


def test_mask_small_values_iwc():
    name = "iwc_lol"
    data = ma.array([[0, 1], [3, 6], [5, 8]])
    data = plt.mask_small_values(data, name)
    compare = ma.array([[0, 1], [3, 6], [5, 8]])
    compare[0, 0] = ma.masked
    testing.assert_array_almost_equal(data.mask, compare.mask)


def test_mask_small_values_iwc_mask():
    name = "iwc_lol"
    data = ma.array([[0, 0.00000001], [3, 6], [5, 8]])
    data = plt.mask_small_values(data, name)
    compare = ma.array([[0, -0.000001], [3, 6], [5, 8]])
    compare[0, 0] = ma.masked
    compare[0, 1] = ma.masked
    testing.assert_array_almost_equal(data.mask, compare.mask)


def test_mask_small_values():
    name = "cf_lol"
    data = ma.array([[0, 1], [3, 6], [5, 8]])
    data = plt.mask_small_values(data, name)
    compare = ma.array([[0, 1], [3, 6], [5, 8]])
    compare[0, 0] = ma.masked
    testing.assert_array_almost_equal(data.mask, compare.mask)


def test_mask_small_values_mask():
    name = "cf_lol"
    data = ma.array([[0, -0.000001], [3, 6], [5, 8]])
    data_mask = plt.mask_small_values(data, name)
    compare = ma.array([[0, -0.000001], [3, 6], [5, 8]])
    compare[0, 0] = ma.masked
    compare[0, 1] = ma.masked
    testing.assert_array_equal(data_mask.mask, compare.mask)


def test_reshape_1d2nd():
    oned = np.array([1, 2, 3, 4])
    twod = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
    compare = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    x = plt.reshape_1d2nd(oned, twod)
    testing.assert_array_almost_equal(x, compare)


def test_create_segment_values():
    model_mask = np.array([[0, 1, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]], dtype=bool)
    obs_mask = np.array([[0, 0, 0, 0], [0, 0, 1, 0], [1, 0, 1, 1]], dtype=bool)
    x, _ = plt.create_segment_values([model_mask, obs_mask])
    compare = np.array([[2, 3, 3, 2], [2, 2, 0, 2], [0, 2, 1, 1]])
    testing.assert_array_almost_equal(x, compare)


def test_rolling_mean():
    data = np.ma.array([1, 2, 7, 4, 2, 3, 8, 5])
    x = plt.rolling_mean(data, 2)
    compare = np.array([1.5, 4.5, 5.5, 3, 2.5, 5.5, 6.5, 5])
    testing.assert_array_almost_equal(x, compare)


def test_rolling_mean_nan():
    data = np.ma.array([1, 2, np.nan, 4, 2, np.nan, 8, 5])
    x = plt.rolling_mean(data, 2)
    compare = np.array([1.5, 2, 4, 3, 2, 8, 6.5, 5])
    testing.assert_array_almost_equal(x, compare)


def test_rolling_mean_mask():
    data = np.ma.array([1, 2, 7, 4, 2, 3, 8, 5])
    data.mask = np.array([0, 0, 1, 0, 1, 0, 0, 1])
    x = plt.rolling_mean(data, 2)
    compare = np.array([1.5, 2, 4, 4, 3, 5.5, 8, np.nan])
    testing.assert_array_almost_equal(x, compare)


def test_rolling_mean_all_mask():
    data = np.ma.array([1, 2, 7, 4, 2, 3, 8, 5])
    data.mask = np.array([0, 1, 1, 1, 1, 0, 0, 1])
    x = plt.rolling_mean(data, 2)
    compare = np.array([1, np.nan, np.nan, np.nan, 3, 5.5, 8, np.nan])
    testing.assert_array_almost_equal(x, compare)


def test_change2one_dim_axes_maskY():
    x = np.ma.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])
    y = np.ma.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    y[1] = np.ma.masked
    data = np.ma.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    x, y, data = plt.change2one_dim_axes(x, y, data)
    compare_x = np.array([1, 2, 3, 4])
    compare_y = np.array([1, 2, 3, 4, 5])
    testing.assert_array_almost_equal(x, compare_x)
    testing.assert_array_almost_equal(y, compare_y)


def test_change2one_dim_axes_maskX():
    x = np.ma.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])
    y = np.ma.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    x[1] = np.ma.masked
    data = np.ma.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    x, y, data = plt.change2one_dim_axes(x, y, data)
    compare_x = np.array([1, 2, 3, 4])
    compare_y = np.array([1, 2, 3, 4, 5])
    testing.assert_array_almost_equal(x, compare_x)
    testing.assert_array_almost_equal(y, compare_y)


def test_change2one_dim_axes():
    x = np.ma.array([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])
    compare_x = np.copy(x)
    y = np.ma.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    compare_y = np.copy(y)
    data = np.ma.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    x, y, data = plt.change2one_dim_axes(x, y, data)
    testing.assert_array_almost_equal(x, compare_x)
    testing.assert_array_almost_equal(y, compare_y)
