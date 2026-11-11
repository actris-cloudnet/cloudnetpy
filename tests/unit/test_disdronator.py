import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from cloudnetpy.disdronator.process import DisdroL1, process_l2


def test_process_l2_with_interval_zero():
    diameter = np.array([0.5, 1.0, 1.5, 2.0])
    diameter_bins = np.array([0.0, 0.75, 1.25, 1.75, 2.5])
    diameter_spread = np.array([0.25, 0.25, 0.25, 0.25])
    velocity = np.array([1.0, 2.0, 3.0, 4.0])
    velocity_bins = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    velocity_spread = np.array([0.5, 0.5, 0.5, 0.5])
    time = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
        dtype="datetime64[s]",
    )
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
    assert_array_equal(
        l2.time,
        np.array(["2024-01-01T00:00:00", "2024-01-01T00:02:00"], dtype="datetime64[s]"),
    )
    assert_allclose(l2.interval, [1.0, 1.0])
    assert_allclose(l2.data_raw, [[1, 2, 3, 4], [9, 10, 11, 12]])
    assert l2.rain_rate.shape == (2,)


def test_process_l2_interval_differs_from_time_step():
    # Bucharest-style data: records every 60 s but the instrument interval is 5 s.
    diameter = np.array([0.5, 1.0, 2.0, 3.0])
    diameter_bins = np.array([0.25, 0.75, 1.5, 2.5, 3.5])
    diameter_spread = np.array([0.5, 0.75, 1.0, 1.0])
    velocity = np.array([2.0, 4.0, 6.5, 8.5])
    velocity_bins = np.array([1.0, 3.0, 5.25, 7.5, 9.5])
    velocity_spread = np.array([2.0, 2.25, 2.25, 2.0])
    time = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:01:00", "2024-01-01T00:02:00"],
        dtype="datetime64[s]",
    )
    interval = np.array([5, 5, 5])
    area_nom = 5000.0  # RD-80 nominal sampling area, mm2
    data_raw = np.array([[40, 8, 2, 1]] * 3)
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
    assert_allclose(l2.rain_rate, 4.222301, rtol=1e-5)  # mm/h
    # Accumulation integrates rain_rate over the actual 60 s time step,
    # not over the 5 s instrument interval.
    assert_allclose(l2.rain_accum, [0.070372, 0.140743, 0.211115], rtol=1e-5)  # mm
