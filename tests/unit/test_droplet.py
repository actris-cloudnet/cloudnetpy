import numpy as np
import numpy.ma as ma
import pytest
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
                             self.threshold) == 4
