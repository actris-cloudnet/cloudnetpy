import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from cloudnetpy.instruments.rpg import _validate_date


def make_object():
    day = datetime.date(2026, 8, 1)
    start = (day - datetime.date(2001, 1, 1)).days * 86400
    obj = SimpleNamespace(
        data={
            "time": np.array(
                [start - 1, start, start + 1, start + 86399, start + 86400]
            ),
            "time_ms": np.array([999, 0, 25, 999, 0]),
            "Zh": np.ma.array(np.arange(10).reshape(5, 2), mask=False),
            "elevation": np.arange(5),
        }
    )
    obj.data["Zh"].mask[2, 0] = True
    return day, obj


def test_boundary_selection_is_half_open_and_preserves_aligned_data():
    day, obj = make_object()
    original = obj.data["Zh"].copy()
    _validate_date(obj, day)
    np.testing.assert_array_equal(obj.data["elevation"], [1, 2, 3])
    np.testing.assert_array_equal(obj.data["time_ms"], [0, 25, 999])
    np.testing.assert_array_equal(obj.data["Zh"].data, original.data[1:4])
    np.testing.assert_array_equal(obj.data["Zh"].mask, original.mask[1:4])


def test_neighbor_days_partition_samples_without_duplication():
    day, _ = make_object()
    counts = []
    for offset in [-1, 0, 1]:
        _, obj = make_object()
        _validate_date(obj, day + datetime.timedelta(days=offset))
        counts.append(len(obj.data["time"]))
    assert counts == [1, 3, 1]


def test_outside_day_rejected():
    day, obj = make_object()
    with pytest.raises(ValueError, match="no time stamps"):
        _validate_date(obj, day + datetime.timedelta(days=2))


def test_without_millisecond_field():
    day, obj = make_object()
    del obj.data["time_ms"]
    _validate_date(obj, day)
    assert len(obj.data["time"]) == 3
