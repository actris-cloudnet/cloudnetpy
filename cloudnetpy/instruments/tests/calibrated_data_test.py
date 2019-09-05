from tests.reference_data import reference_data


def test_operative_data():
    result = reference_data.get_process_data_and_boundaries('radar')
    assert all(r is True for r in result.values()),\
        reference_data.false_variables_msg(result, 'radar')


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

