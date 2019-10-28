import numpy as np
import numpy.ma as ma
import pytest
from numpy.testing import assert_array_equal
from cloudnetpy.categorize import droplet


class TestIndBase:

    x = np.array([0, 0.5, 1, -99, 4, 8, 5])
    mx = ma.masked_array(x, mask=[0, 0, 0, 1, 0, 0, 0])
    dx = ma.diff(mx).filled(0)
    peak_position = 5
    search_range = 4
    threshold = 2

    def test_ind_base_0(self):
        assert droplet.ind_base(self.dx, self.peak_position, self.search_range,
                                self.threshold) == 4

    def test_ind_base_1(self):
        search_range = 100
        assert droplet.ind_base(self.dx, self.peak_position, search_range,
                                self.threshold) == 4

    def test_ind_base_2(self):
        huge_threshold = 5000
        assert droplet.ind_base(self.dx, self.peak_position, self.search_range,
                                huge_threshold) == 1

    def test_ind_base_3(self):
        small_threshold = 1.01
        assert droplet.ind_base(self.dx, self.peak_position, self.search_range,
                                small_threshold) == 4

    def test_ind_base_4(self):
        mx = ma.masked_array(self.x, mask=[1, 0, 1, 1, 1, 0, 0])
        dx = ma.diff(mx).filled(0)
        with pytest.raises(IndexError):
            droplet.ind_base(dx, self.peak_position, self.search_range,
                             self.threshold)


class TestIndTop:

    x = np.array([1, 3, 8, 4, -99, 1, 0.5, 0])
    mx = ma.masked_array(x, mask=[0, 0, 0, 0, 1, 0, 0, 0])
    dx = ma.diff(mx).filled(0)
    peak_position = 2
    n_prof = x.shape[0]
    search_range = 5
    threshold = 2

    def test_ind_top_0(self):
        assert droplet.ind_top(self.dx, self.peak_position, self.n_prof,
                               self.search_range, self.threshold) == 3

    def test_ind_top_1(self):
        search_range = 100
        assert droplet.ind_top(self.dx, self.peak_position, self.n_prof,
                               search_range, self.threshold) == 3

    def test_ind_top_2(self):
        huge_threshold = 5000
        assert droplet.ind_top(self.dx, self.peak_position, self.n_prof,
                               self.search_range, huge_threshold) == 7

    def test_ind_top_3(self):
        small_threshold = 1.01
        assert droplet.ind_top(self.dx, self.peak_position, self.n_prof,
                               self.search_range, small_threshold) == 3

    def test_ind_top_4(self):
        mx = ma.masked_array(self.x, mask=[1, 0, 1, 0, 1, 0, 1, 0])
        dx = ma.diff(mx).filled(0)
        with pytest.raises(IndexError):
            droplet.ind_top(dx, self.peak_position, self.n_prof,
                            self.search_range, self.threshold)


def test_find_strong_peaks():
    data = np.array([[0, 0, 100, 1e6, 100, 0, 0],
                     [0, 100, 1e6, 100, 0, 0, 0]])
    threshold = 1e3
    peaks = droplet._find_strong_peaks(data, threshold)
    assert_array_equal(peaks, ([0, 1], [3, 2]))


def test_intepolate_lwp():
    class Obs:
        def __init__(self):
            self.time = np.arange(11)
            self.lwp_orig = np.linspace(1, 11, 11)
            self.lwp = ma.masked_where(self.lwp_orig % 2 == 0, self.lwp_orig)
    obs = Obs()
    lwp_intepolated = droplet._interpolate_lwp(obs)
    assert_array_equal(obs.lwp_orig, lwp_intepolated)
