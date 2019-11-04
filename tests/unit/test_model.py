import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
import pytest
from cloudnetpy.categorize import model


@pytest.mark.parametrize("input, result", [
    ('this_is_a_ecmwf_model_file.nc', 'ecmwf'),
    ('a_gdasXYZ_model_file.nc', 'gdas'),
    ('an_unknown.nc', '')

])
def test_find_model_type(input, result):
    assert model._find_model_type(input) == result


def test_calc_mean_height():
    height = np.array([[0, 1, 2, 3, 4],
                       [0.2, 1.2, 2.2, 3.2, 4.2],
                       [-0.2, 0.8, 1.8, 2.8, 3.8]])
    result = np.array([0, 1, 2, 3, 4])
    assert_array_equal(model._calc_mean_height(height), result)
