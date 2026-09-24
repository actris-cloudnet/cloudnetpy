import numpy as np
from numpy.testing import assert_allclose

from cloudnetpy.disdronator.process import DisdroL1, process_l2


def test_process_l2_with_interval_zero():
    diameter = np.array([0.5, 1.0, 1.5, 2.0])
    diameter_bins = np.array([0.0, 0.75, 1.25, 1.75, 2.5])
    diameter_spread = np.array([0.25, 0.25, 0.25, 0.25])
    velocity = np.array([1.0, 2.0, 3.0, 4.0])
    velocity_bins = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    velocity_spread = np.array([0.5, 0.5, 0.5, 0.5])
    time = np.array([0.0, 1.0, 2.0])
    interval = np.array([1.0, 0.0, 1.0])
    area_nom = 50.0
    data_raw = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    l1 = DisdroL1(
        diameter=diameter,
        diameter_bins=diameter_bins,
        diameter_spread=diameter_spread,
        velocity=velocity,
        velocity_bins=velocity_bins,
        velocity_spread=velocity_spread,
        time=time,
        interval=interval,
        area_nom=area_nom,
        area_eff=None,
        data_raw=data_raw,
        altitude=None,
    )
    l2 = process_l2(l1)
    assert_allclose(l2.time, [0.0, 2.0])
    assert_allclose(l2.interval, [1.0, 1.0])
    assert_allclose(l2.data_raw, [[1, 2, 3, 4], [9, 10, 11, 12]])
    assert l2.rain_rate.shape == (2,)
