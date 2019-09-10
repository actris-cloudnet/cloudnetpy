from tests.data_quality import data_quality_test
from tests.test import LoggingHandler


FILE_TYPES = ['radar', 'ceilo']


def test_operative_data():
    for type in FILE_TYPES:
        result = data_quality_test.get_process_data_and_boundaries(type)
        assert all(r for r in result.values()),\
            LoggingHandler.fill_log(data_quality_test.false_variables_msg(result), type)


def test_operative_peaks():
    assert True


def test_reference_data():
    assert True


def test_reference_mean():
    assert True


def test_reference_median():
    assert True


def test_reference_std():
    assert True


def test_reference_N():
    assert True


def test_reference_distribution():
    assert True

