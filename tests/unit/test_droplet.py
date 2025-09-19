import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_array_equal

from cloudnetpy.categorize import droplet


class TestIndBase:
    x = ma.array([0, 0.5, 1, -99, 4, 8, 5])
    mx: ma.MaskedArray = ma.masked_array(x, mask=[0, 0, 0, 1, 0, 0, 0])
    diffu = np.diff(mx)
    assert isinstance(diffu, ma.MaskedArray)
    dx = diffu.filled(0)
    peak_position = 5
    search_range = 4
    threshold = 2

    def test_ind_base_0(self):
        assert (
            droplet.ind_base(
                self.dx,
                self.peak_position,
                self.search_range,
                self.threshold,
            )
            == 4
        )

    def test_ind_base_1(self):
        search_range = 100
        assert (
            droplet.ind_base(self.dx, self.peak_position, search_range, self.threshold)
            == 4
        )

    def test_ind_base_2(self):
        huge_threshold = 5000
        assert (
            droplet.ind_base(
                self.dx,
                self.peak_position,
                self.search_range,
                huge_threshold,
            )
            == 1
        )

    def test_ind_base_3(self):
        small_threshold = 1.01
        assert (
            droplet.ind_base(
                self.dx,
                self.peak_position,
                self.search_range,
                small_threshold,
            )
            == 4
        )

    def test_ind_base_4(self):
        mx: ma.MaskedArray = ma.masked_array(self.x, mask=[1, 0, 1, 1, 1, 0, 0])
        diffu = ma.diff(mx)
        assert isinstance(diffu, ma.MaskedArray)
        dx = diffu.filled(0)
        with pytest.raises(IndexError):
            droplet.ind_base(dx, self.peak_position, self.search_range, self.threshold)


class TestIndTop:
    x = np.array([1, 3, 8, 4, -99, 1, 0.5, 0])
    mx: ma.MaskedArray = ma.masked_array(x, mask=[0, 0, 0, 0, 1, 0, 0, 0])
    diffu = np.diff(mx)
    assert isinstance(diffu, ma.MaskedArray)
    dx = diffu.filled(0)
    peak_position = 2
    n_prof = x.shape[0]
    search_range = 5
    threshold = 2

    def test_ind_top_0(self):
        assert (
            droplet.ind_top(
                self.dx,
                self.peak_position,
                self.n_prof,
                self.search_range,
                self.threshold,
            )
            == 3
        )

    def test_ind_top_1(self):
        search_range = 100
        assert (
            droplet.ind_top(
                self.dx,
                self.peak_position,
                self.n_prof,
                search_range,
                self.threshold,
            )
            == 3
        )

    def test_ind_top_2(self):
        huge_threshold = 5000
        assert (
            droplet.ind_top(
                self.dx,
                self.peak_position,
                self.n_prof,
                self.search_range,
                huge_threshold,
            )
            == 7
        )

    def test_ind_top_3(self):
        small_threshold = 1.01
        assert (
            droplet.ind_top(
                self.dx,
                self.peak_position,
                self.n_prof,
                self.search_range,
                small_threshold,
            )
            == 3
        )

    def test_ind_top_4(self):
        mx: ma.MaskedArray = ma.masked_array(self.x, mask=[1, 0, 1, 0, 1, 0, 1, 0])
        diffu = ma.diff(mx)
        assert isinstance(diffu, ma.MaskedArray)
        dx = diffu.filled(0)
        with pytest.raises(IndexError):
            droplet.ind_top(
                dx,
                self.peak_position,
                self.n_prof,
                self.search_range,
                self.threshold,
            )


def test_find_strong_peaks():
    data = np.array([[0, 0, 100, 1e6, 100, 0, 0], [0, 100, 1e6, 100, 0, 0, 0]])
    threshold = 1e3
    peaks = droplet._find_strong_peaks(data, threshold)
    assert_array_equal(peaks, ([0, 1], [3, 2]))


def test_interpolate_lwp():
    class Obs:
        def __init__(self):
            self.time = np.arange(11)
            self.lwp_orig = np.linspace(1, 11, 11)
            self.lwp = ma.masked_where(self.lwp_orig % 2 == 0, self.lwp_orig)

    obs = Obs()
    lwp_interpolated = droplet.interpolate_lwp(obs)  # type: ignore
    assert_array_equal(obs.lwp_orig, lwp_interpolated)


def test_interpolate_lwp_masked():
    class Obs:
        def __init__(self):
            self.time = np.arange(5)
            self.lwp = ma.array([1, 2, 3, 4, 5], mask=True)

    obs = Obs()
    lwp_interpolated = droplet.interpolate_lwp(obs)  # type: ignore
    assert_array_equal(
        np.zeros(
            5,
        ),
        lwp_interpolated,
    )


@pytest.mark.parametrize(
    "is_freezing, top_above, result",
    [
        ([0, 0, 1, 1, 1, 1], 2, 4),
        ([1, 1, 1, 1, 1, 1], 2, 2),
        ([1, 1, 1, 1, 1, 1], 10, 5),
    ],
)
def test_find_ind_above_top(is_freezing, top_above, result):
    assert droplet._find_ind_above_top(is_freezing, top_above) == result


def test_correct_liquid_top():
    class Obs:
        def __init__(self):
            self.height = np.arange(11)
            self.z: ma.MaskedArray = ma.masked_array(
                np.random.random((3, 10)),
                mask=[
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                ],
            )  # Here one masked value

    is_freezing = np.array(
        [
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        ],
    )

    liquid = np.array(
        [
            [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        ],
        dtype=bool,
    )

    obs = Obs()
    corrected = droplet.correct_liquid_top(obs, liquid, is_freezing, limit=100)  # type: ignore
    liquid[2, :] = [0, 0, 0, 1, 1, 1, 1, 1, 0, 0]
    assert_array_equal(corrected, liquid)


def test_find_liquid():
    class ClassData:
        def __init__(self):
            self.height = np.arange(6) * 100
            self.time = np.arange(3)
            self.lwp = ma.array(range(3))
            self.beta = ma.array(
                [
                    [1e-8, 1e-8, 2e-6, 1e-3, 5e-6, 1e-8],
                    [1e-8, 1e-8, 1e-5, 1e-3, 1e-5, 1e-8],
                    [1e-8, 1e-8, 1e-5, 1e-3, 1e-5, 1e-8],
                ],
                mask=[
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
            )

    is_liquid = np.array([[0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0], [0, 0, 1, 1, 1, 0]])

    obs = ClassData()
    assert_array_equal(is_liquid, droplet.find_liquid(obs))  # type: ignore
