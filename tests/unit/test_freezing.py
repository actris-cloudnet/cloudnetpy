import numpy as np
from numpy.testing import assert_array_equal
from cloudnetpy.categorize import freezing
from cloudnetpy.constants import T0


def test_find_t0_alt():
    temperature = np.array([[290, 280, T0, 260],
                           [320, T0+10, T0-10, 220],
                           [240, 230, 220, 210]])
    height = np.array([10, 20, 30, 40])
    res = [30, 25, 10]
    cnet = freezing.find_t0_alt(temperature, height)
    assert_array_equal(cnet, res)
