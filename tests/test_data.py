import pytest
import numpy as np
import numpy.ma as ma
import logging
from tests.quality_control import get_file_keys

keys = get_file_keys()


@pytest.mark.parametrize('data', keys, indirect=True)
def test_min(data):
    for array, key, limit in zip(data.variables, data.keys, data.min):
        assert ma.min(array) >= limit, logging.warning(f"Too small value in {key}: {ma.min(array)} ({limit})")


@pytest.mark.parametrize('data', keys, indirect=True)
def test_max(data):
    for array, key, limit in zip(data.variables, data.keys, data.max):
        assert ma.max(array) <= limit, logging.warning(f"Too large value in {key}: {ma.max(array)} ({limit})")
