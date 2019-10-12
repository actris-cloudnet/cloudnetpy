
def test_variable_keys(missing_variables):
    assert not missing_variables


def test_variable_values(data):
    assert not data.bad_values
