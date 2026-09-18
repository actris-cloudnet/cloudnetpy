import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_array_equal

from cloudnetpy.categorize import melting


@pytest.mark.parametrize(
    "model, result",
    [
        ("gdas1", (-8, 6)),
        ("ecmwf", (-4, 3)),
        ("some_icon_version", (-4, 3)),
        ("some_harmonie_version", (-4, 3)),
    ],
)
def test_find_model_temperature_range(model, result):
    assert melting._find_model_temperature_range(model) == result


@pytest.mark.parametrize(
    "t_prof, t_range, result",
    [
        (np.array([500, 200]), (-10, 10), []),
        (np.array([500, 400]), (-10, 10), []),
        (np.array([220, 210, 200]), (-10, 10), []),
        (np.array([272, 100]), (-10, 10), [0]),
        (np.array([500, 272]), (-10, 10), [1]),
        (np.array([280, 270, 260]), (-5, 2), [1]),
        (np.linspace(275, 270, 10), (-4, -3), [9]),
        (np.array([300, 290, 280, 270, 260, 250, 240]), (-5, 5), [3]),
        (np.array([274, 272]), (-10, 10), [0, 1]),
        (np.array([274, 272, 100, 50]), (-10, 10), [0, 1]),
        (np.array([270, 275, 260, 250, 240]), (-10, 10), [0, 1]),
        (np.array([500, 400, 274, 272]), (-10, 10), [2, 3]),
        (np.array([500, 400, 274, 272, 100, 50]), (-10, 10), [2, 3]),
        (np.array([300, 290, 280, 270, 260, 250, 240]), (-10, 10), [2, 3]),
        (np.array([290, 280, 270, 275, 260, 250, 240]), (-10, 10), [1, 2, 3]),
        (np.array([500, 400, 273, 420, 272, 200]), (-10, 10), [2, 3, 4]),
        (np.array([200, 272, 420, 274, 273, 150]), (-10, 10), [1, 2, 3, 4]),
        (
            np.array([200, 272, 420, 274, 100, 273, 100]),
            (-10, 10),
            [1, 2, 3, 4, 5],
        ),
    ],
)
def test_get_temp_indices(t_prof, t_range, result):
    indices = melting._get_temp_indices(t_prof, t_range)
    assert_array_equal(indices, result)


@pytest.mark.parametrize(
    "t_prof, result",
    [
        # Monotonic profile is not affected
        (np.array([275, 274, 272, 270, 268]), [0, 1, 2, 3]),
        # Subzero inversion far above the 0 C level is excluded
        (np.array([274, 272, 268, 268, 268, 271, 268]), [0, 1]),
        # Inversion reaching 0 C is included
        (np.array([274, 272, 268, 268, 268, 274, 268]), [0, 1, 2, 3, 4, 5]),
        # All subzero
        (np.array([270, 271, 268, 268, 270, 271, 268]), []),
    ],
)
def test_get_temp_indices_with_height(t_prof, result):
    height = np.arange(len(t_prof)) * 400.0
    indices = melting._get_temp_indices(t_prof, (-4, 3), height)
    assert_array_equal(indices, result)


class Obs:
    def __init__(self):
        self.tw = np.tile(np.linspace(275, 270, 10), (5, 1))
        self.tw[:, -1] = 250
        self.ldr = ma.array(np.tile([1, 1, 1, 3, 150, 3, 1, 1, 1, 1], (5, 1)))
        self.v = ma.array(np.tile([-1, -1, -4, -2, -2, -1, 0, 0, 0, 0], (5, 1)))
        self.z = ma.array(np.tile([1, 1, 1, 1, 1, 3, 1, 1, 1, 1], (5, 1)))
        self.model_type = "ecmwf"
        self.height = np.arange(10) * 100.0


def test_find_melting_layer():
    obs = Obs()
    layer = melting.find_melting_layer(obs, smooth=False)  # type: ignore
    result = np.tile([0, 0, 0, 1, 1, 0, 0, 0, 0, 0], (5, 1))
    assert_array_equal(layer, result)


@pytest.mark.parametrize(
    "ldr_mask, v_mask, result",
    [
        ([0, 0, 0, 0], [0, 0, 0, 0], True),
        ([1, 1, 0, 0], [0, 0, 0, 0], True),
        ([1, 1, 1, 0], [0, 0, 0, 0], False),
        ([1, 1, 1, 1], [0, 0, 0, 0], False),
        ([1, 1, 1, 0], [1, 1, 0, 0], True),
    ],
)
def test_has_ldr_coverage(ldr_mask, v_mask, result):
    ldr = ma.array([-30.0, -30, -30, -30], mask=ldr_mask)
    v = ma.array([-1.0, -1, -1, -1], mask=v_mask)
    assert melting._has_ldr_coverage(ldr, v) is result


def test_has_ldr_coverage_without_ldr():
    assert melting._has_ldr_coverage(None, ma.array([-1.0, -1])) is False
