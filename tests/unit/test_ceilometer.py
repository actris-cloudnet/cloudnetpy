import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
from cloudnetpy.instruments import ceilometer


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


def test_reset_low_values_above_saturation():
    beta = ma.array([[0, 10, 1e-6, 3],
                     [0, 0, 0, 0.1],
                     [0, 0.6, 1.2, 1e-8]],
                    mask=[[0, 0, 0, 0],
                          [0, 0, 0, 0],
                          [0, 0, 0, 0]])
    noise = 1e-3
    saturated = [1, 0, 1]
    result = ma.array([[0, 10, 1e-6, 3],
                       [0, 0, 0, 0.1],
                       [0, 0.6, 1.2, 1e-8]],
                      mask=[[0, 0, 1, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1]])
    call = ceilometer._reset_low_values_above_saturation(beta, saturated, noise)
    assert_array_equal(call.data, result.data)
    assert_array_equal(call.mask, result.mask)



