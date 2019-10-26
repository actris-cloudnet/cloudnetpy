import numpy as np
from numpy.testing import assert_array_equal
from cloudnetpy.categorize import datasource


class FakeProfileDataSource(datasource.ProfileDataSource):
    def __init__(self, value, units):
        self.variables = {'height': Variable(value, units)}
        self.height = self._get_height()


class Variable:
    def __init__(self, value, units):
        self.value = value
        self.units = units

    def __getitem__(self, ind):
        return self.value[ind]


def test_height():
    obj = FakeProfileDataSource(np.array([1, 2, 3]), 'km')
    assert_array_equal(obj.height, [1000, 2000, 3000])
