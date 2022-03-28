import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_equal
from cloudnetpy.categorize import insects


class Obs:
    def __init__(self):
        self.radar_type = "MIRA-35"
        self.lwp = np.array([1, 2, 3, 4])
        self.v = ma.array([[0, 1, 1, 0], [0, 1, -99, 0]], mask=[[0, 0, 0, 0], [1, 0, 1, 0]])
        self.is_rain = np.array([0, 0, 1])


def test_calc_prob_from_ldr():
    prob = {"z": 0.5, "temp_loose": 0.5, "ldr": 0.5, "temp_strict": 0.5}
    assert insects._calc_prob_from_ldr(prob) == 0.5**2


def test_calc_prob_from_all():
    prob = {"z": 0.5, "temp_strict": 0.5, "v": 0.5, "width": 0.5, "z_weak": 0.5}
    assert insects._calc_prob_from_all(prob) == 0.5**4


def test_adjust_for_radar():
    prob = {"lwp": np.array([0.5, 0.5, 0.5, 0.5])}
    prob_from_others = np.array([1.0, 1.0, 1.0, 1.0])
    obs = Obs()
    assert_array_equal(
        insects._adjust_for_radar(obs, prob, prob_from_others), np.array([0.5, 0.5, 0.5, 0.5])
    )


def test_fill_missing_pixels():
    prob_from_ldr = np.array(
        [[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]]
    )
    prob_from_others = np.array(
        [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
    )
    result = np.array([[0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5], [0.5, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5]])
    assert_array_equal(insects._fill_missing_pixels(prob_from_ldr, prob_from_others), result)


def test_get_smoothed_v():
    obs = Obs()
    result = ma.array([[0, 1, 1, 0], [0, 1, -99, 0]], mask=[[0, 0, 0, 0], [1, 0, 1, 0]])
    assert_array_equal(insects._get_smoothed_v(obs, sigma=(0, 0)), result)


def test_screen_insets():
    obs = Obs()
    insect_prob = np.ones((3, 4))
    insect_prob_no_ldr = np.ones((3, 4)) * 0.5
    melting_layer = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    liquid_layers = np.array([[0, 0, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]])
    result = np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]])
    screened_prob = insects._screen_insects(
        insect_prob, insect_prob_no_ldr, melting_layer, liquid_layers, obs
    )
    assert_array_equal(screened_prob, result)
