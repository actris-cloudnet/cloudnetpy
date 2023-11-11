from datetime import datetime, timezone

import numpy as np
import pytest
from numpy import ma, testing

from cloudnetpy.model_evaluation.products.product_resampling import ObservationManager
from cloudnetpy.products.product_tools import CategorizeBits

PRODUCT = "iwc"


class CatBits:
    def __init__(self) -> None:
        self.category_bits = {
            "droplet": np.asarray([[1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]], dtype=bool),
            "falling": np.asarray([[0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=bool),
            "cold": np.asarray([[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 0, 1]], dtype=bool),
            "melting": np.asarray([[1, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=bool),
            "aerosol": np.asarray([[1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=bool),
            "insect": np.asarray([[1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=bool),
        }


def test_get_date(obs_file) -> None:
    obj = ObservationManager(PRODUCT, str(obs_file))
    date = datetime(2019, 5, 23, 0, 0, 0, tzinfo=timezone.utc)
    assert obj._get_date() == date


@pytest.mark.parametrize("key", ["iwc", "lwc", "cf"])
def test_generate_product(key, obs_file) -> None:
    obj = ObservationManager(key, str(obs_file))
    obj._generate_product()
    assert key in obj.data


def test_add_height(obs_file) -> None:
    obj = ObservationManager(PRODUCT, str(obs_file))
    obj._generate_product()
    assert "height" in obj.data


def test_generate_cf(obs_file) -> None:
    obj = ObservationManager("cf", str(obs_file))
    x = obj._generate_cf()
    compare = ma.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    )
    compare[~obj._rain_index(), :] = ma.masked
    testing.assert_array_almost_equal(compare, x)


def test_basic_cloud_mask(obs_file) -> None:
    cat = CategorizeBits(str(obs_file))
    obj = ObservationManager("cf", str(obs_file))
    x = obj._classify_basic_mask(cat.category_bits)
    compare = np.array(
        [
            [0, 1, 2, 0],
            [2, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 6, 6],
            [7, 2, 0, 7],
        ],
    )
    testing.assert_array_almost_equal(x, compare)


def test_mask_cloud_bits(obs_file) -> None:
    cat = CategorizeBits(str(obs_file))
    obj = ObservationManager("cf", str(obs_file))
    mask = obj._classify_basic_mask(cat.category_bits)
    compare = obj._mask_cloud_bits(mask)
    x = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
    )
    testing.assert_array_almost_equal(x, compare)


def test_basic_cloud_mask_all_values(obs_file) -> None:
    cat = CatBits()
    obj = ObservationManager("cf", str(obs_file))
    x = obj._classify_basic_mask(cat.category_bits)
    compare = np.array([[8, 7, 6, 1, 3, 1], [0, 1, 7, 5, 2, 4]])
    testing.assert_array_almost_equal(x, compare)


def test_mask_cloud_bits_all_values(obs_file) -> None:
    cat = CatBits()
    obj = ObservationManager("cf", str(obs_file))
    mask = obj._classify_basic_mask(cat.category_bits)
    x = obj._mask_cloud_bits(mask)
    compare = np.array([[0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1]])
    testing.assert_array_almost_equal(x, compare)


def test_check_rainrate(obs_file) -> None:
    obj = ObservationManager("cf", str(obs_file))
    x = obj._check_rainrate()
    assert x is True


def test_get_rainrate_threshold(obs_file) -> None:
    obj = ObservationManager("cf", str(obs_file))
    x = obj._get_rainrate_threshold()
    assert x == 8


def test_rain_index(obs_file) -> None:
    obj = ObservationManager("cf", str(obs_file))
    x = obj._rain_index()
    compare = np.array([0, 0, 0, 1, 0, 1], dtype=bool)
    testing.assert_array_almost_equal(x, compare)


@pytest.mark.parametrize("key", ["iwc", "iwc_att", "iwc_rain"])
def test_generate_iwc_masks(key, obs_file) -> None:
    obj = ObservationManager(PRODUCT, str(obs_file))
    obj._generate_iwc_masks()
    assert key in obj.data


def test_get_rain_iwc(obs_file) -> None:
    obj = ObservationManager("iwc", str(obs_file))
    iwc_status = obj.getvar("iwc_retrieval_status")
    x = np.zeros(iwc_status.shape)
    x[iwc_status == 5] = 1
    x = np.any(x, axis=1)
    obj._get_rain_iwc(iwc_status[:])
    compare = obj.data["iwc_rain"]
    testing.assert_array_almost_equal(x, compare[:])


def test_mask_iwc_att(obs_file) -> None:
    obj = ObservationManager("iwc", str(obs_file))
    iwc = obj.getvar("iwc")
    iwc_status = obj.getvar("iwc_retrieval_status")
    x = ma.copy(iwc)
    obj._mask_iwc_att(iwc, iwc_status)
    compare = obj.data["iwc_att"]
    x[iwc_status > 3] = ma.masked
    testing.assert_array_almost_equal(x, compare[:])
