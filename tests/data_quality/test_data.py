from tests.utils import log_errors


@log_errors
def test_min(too_small_values):
    assert not too_small_values


@log_errors
def test_max(too_large_values):
    assert not too_large_values


@log_errors
def test_invalid(invalid_values):
    assert not invalid_values


