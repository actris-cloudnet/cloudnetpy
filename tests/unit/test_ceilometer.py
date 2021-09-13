import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal, assert_array_almost_equal
from cloudnetpy.instruments import ceilometer
import pytest


class TestNoisyData:

    @pytest.fixture(autouse=True)
    def run_before_and_after_tests(self):
        data = {'backscatter': np.array([[1, 2, 3],
                                         [1, 2, 3]])}
        range_squared = np.array([1, 2, 3])
        snr_limit = 1
        noise_params = (1, 1, 1, (1, 1))
        self.noisy_data = ceilometer.NoisyData(data, range_squared, noise_params, snr_limit)
        yield

    def test_remove_noise(self):
        noise = np.array([1.1, -1.1])
        array = ma.array([[1, 2, 3],
                          [1, 2, 3]], mask=False)
        expected = ma.array([[1, 2, 3],
                             [1, 2, 3]], mask=[[1, 0, 0],
                                               [1, 0, 0]])
        screened_array = self.noisy_data._remove_noise(array, noise)
        assert_array_equal(screened_array.mask, expected.mask)

    def test_calc_range_uncorrected(self):
        beta = np.array([[1, 2, 3],
                         [1, 2, 3]])
        expected = np.ones((2, 3))
        assert_array_equal(self.noisy_data._calc_range_uncorrected(beta), expected)

    def test_calc_range_corrected(self):
        beta = np.ones((2, 3))
        expected = np.array([[1, 2, 3],
                             [1, 2, 3]])
        assert_array_equal(self.noisy_data._calc_range_corrected(beta), expected)

    def test_reset_low_values_above_saturation(self):
        beta = ma.array([[0, 10, 1e-6, 3],
                         [0, 0, 0, 0.1],
                         [0, 0.6, 1.2, 1e-8]])
        noise = 1e-3
        self.noisy_data._is_saturation = np.array([1, 0, 1])
        expected = ma.array([[0, 10, 1e-6, 3],
                             [0, 0, 0, 0.1],
                             [0, 0.6, 1.2, 1e-8]], mask=[[0, 0, 1, 0],
                                                         [0, 0, 0, 0],
                                                         [0, 0, 0, 1]])
        result = self.noisy_data._reset_low_values_above_saturation(beta, noise)
        assert_array_equal(result.data, expected.data)
        assert_array_equal(result.mask, expected.mask)

    def test_find_saturated_profiles(self):
        self.noisy_data.noise_params = (2, 0.25, 1, (1, 1))
        self.noisy_data.data['backscatter'] = np.array([[0, 10, 1, 1.99],
                                                        [0, 10, 2.1, 1],
                                                        [0, 10, 1, 1]])
        result = [1, 0, 1]
        assert_array_equal(self.noisy_data._find_saturated_profiles(), result)


class TestCeilometer:

    def test_calc_sigma_units(self):
        time = np.linspace(0, 24, 721)  # 2 min resolution
        range_instru = np.arange(0, 1000, 5)  # 5 m resolution
        std_time, std_range = ceilometer._calc_sigma_units(time, range_instru)
        assert_array_almost_equal(std_time, 1)
        assert_array_almost_equal(std_range, 1)

    def test_get_range_squared(self):
        obj = ceilometer.Ceilometer('/foo/bar')
        obj.range = np.array([1000, 2000, 3000])
        result = np.array([1, 4, 9])
        assert_array_equal(obj._get_range_squared(), result)


def test_estimate_clouds_from_beta():
    beta = ma.array([[0, 0, 1e-5, 0],
                     [0, 10, 1e-7, 0],
                     [0, 1e-4, 1e-3, 0]],
                    mask=[[0, 0, 0, 0],
                          [1, 1, 0, 0],
                          [0, 0, 0, 0]])

    cloud_ind = ([0, 1, 2, 2], [2, 1, 1, 2])
    cloud_beta = ([1e-5, 10, 1e-4, 1e-3])
    a, b, c = ceilometer._estimate_clouds_from_beta(beta)
    assert_array_equal(a, cloud_ind)
    assert_array_equal(b, cloud_beta)
    assert_array_equal(c, 1e-6)


def test_estimate_noise_from_top_gates():
    beta = ma.array([[0, 0.4, 0.1, 0.2],
                     [0, 0, 0, 0.1],
                     [0, 0.6, 1.2, 5.3]],
                    mask=[[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [1, 1, 1, 1]])
    noise_min = 0.001
    result = np.array([np.std([0.4, 0.1, 0.2]), np.std([0, 0, 0.1]), noise_min])
    assert_array_equal(ceilometer._estimate_noise_from_top_gates(beta, 3, noise_min),
                       result)
