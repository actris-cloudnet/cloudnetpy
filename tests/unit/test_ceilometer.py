import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal, assert_array_almost_equal
from cloudnetpy.instruments import ceilometer
import pytest


class TestNoisyData:
    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        data = {"beta_raw": np.array([[1, 2, 3], [1, 2, 3]]), "range": np.array([1000, 2000, 3000])}
        noise_params = ceilometer.NoiseParam()
        self.noisy_data = ceilometer.NoisyData(data, noise_params)
        yield

    def test_remove_noise(self):
        noise = np.array([1.1, -1.1])
        array = ma.array([[1, 2, 3], [1, 2, 3]], mask=False)
        expected = ma.array([[1, 2, 3], [1, 2, 3]], mask=[[1, 0, 0], [1, 0, 0]])
        screened_array = self.noisy_data._remove_noise(array, noise, True, snr_limit=1)
        assert_array_equal(screened_array.mask, expected.mask)

    def test_remove_low_values_above_consequent_negatives(self):
        data = ma.array(
            [[2, 0, 0, 0, 0, 4, 0], [3, -1, -2, -3, 0.9, 1.1, 0], [1, 2, 1, 2, 0, 0, 0]], mask=False
        )
        expected = ma.array(
            [[2, 0, 0, 0, 0, 4, 0], [3, -1, -2, -3, 0.9, 1.1, 0], [1, 2, 1, 2, 0, 0, 0]],
            mask=[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0]],
        )

        res = self.noisy_data._mask_low_values_above_consequent_negatives(
            data, n_negatives=2, threshold=1, n_gates=5, n_skip_lowest=0
        )
        expected_indices = np.array([1])
        assert res == expected_indices
        assert_array_equal(data.data, expected.data)
        assert_array_equal(data.mask, expected.mask)

    def test_clean_fog_profiles(self):
        is_fog = np.array([1, 0, 1])
        data = ma.array(
            [[1, 2, 5, 0, 0.8, 1.1, 0], [3, -1, -2, -3, 0.9, 1.1, 0], [10, 2, 0.2, 2, 0, 0, 0]],
            mask=False,
        )
        expected = ma.array(
            [[1, 2, 5, 0, 0.8, 1.1, 0], [3, -1, -2, -3, 0.9, 1.1, 0], [10, 2, 0.2, 2, 0, 0, 0]],
            mask=[[0, 0, 0, 1, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 1, 1, 1]],
        )
        self.noisy_data._clean_fog_profiles(data, is_fog, threshold=1)
        assert_array_equal(data.data, expected.data)
        assert_array_equal(data.mask, expected.mask)

    def test_calc_range_uncorrected(self):
        beta = np.array([[1, 4, 9], [1, 4, 9]])
        expected = np.ones((2, 3))
        self.noisy_data._calc_range_uncorrected(beta)
        assert_array_equal(beta, expected)

    def test_calc_range_corrected(self):
        beta = np.ones((2, 3))
        expected = np.array([[1, 4, 9], [1, 4, 9]])
        self.noisy_data._calc_range_corrected(beta)
        assert_array_equal(beta, expected)

    def test_find_fog_profiles(self):
        self.noisy_data.data["beta_raw"] = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 1, 1.99],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.1e-12, 1e-12, 1e-12],
                [2e-3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 2.1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 1, 20],
            ]
        )
        result = [0, 1, 1, 0]
        assert_array_equal(self.noisy_data._find_fog_profiles(), result)

    def test_get_range_squared(self):
        result = np.array([1, 4, 9])
        assert_array_equal(self.noisy_data._get_range_squared(), result)


class TestCeilometer:
    def test_calc_sigma_units(self):
        time = np.linspace(0, 24, 721)  # 2 min resolution
        range_instru = np.arange(0, 1000, 5)  # 5 m resolution
        std_time, std_range = ceilometer.calc_sigma_units(time, range_instru)
        assert_array_almost_equal(std_time, 1)
        assert_array_almost_equal(std_range, 1)


def test_estimate_clouds_from_beta():
    beta = ma.array(
        [[0, 0, 1e-5, 0], [0, 10, 1e-7, 0], [0, 1e-4, 1e-3, 0]],
        mask=[[0, 0, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
    )

    cloud_ind = ([0, 1, 2, 2], [2, 1, 1, 2])
    cloud_beta = [1e-5, 10, 1e-4, 1e-3]
    a, b, c = ceilometer._estimate_clouds_from_beta(beta)
    assert_array_equal(a, cloud_ind)
    assert_array_equal(b, cloud_beta)
    assert_array_equal(c, 1e-6)


def test_estimate_background_noise():
    beta = ma.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.4, 0.1, 0.2],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6, 1.2, 5.3],
        ],
        mask=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
    )
    result = np.array([np.std([0.1, 0.2]), np.std([0, 0.1]), np.std([0.6, 1.2, 5.3])])
    assert_array_equal(ceilometer._estimate_background_noise(beta), result)
