import pytest
from numpy.testing import assert_array_equal
from cloudnetpy.instruments import rpg


@pytest.fixture
def example_files(tmpdir):
    file_names = ['f.LV1', 'f.txt', 'f.LV0', 'f.lv1', 'g.LV1']
    folder = tmpdir.mkdir('data/')
    for name in file_names:
        with open(folder.join(name), 'wb') as f:
            f.write(b'abc')
    return folder


def test_get_rpg_files(example_files):
    dir_name = example_files.dirname + '/data'
    result = ['/'.join((dir_name, x)) for x in ('f.LV1', 'g.LV1')]
    assert rpg.get_rpg_files(dir_name) == result


class TestReduceHeader:
    n_points = 100
    header = {'a': n_points * [1], 'b': n_points * [2], 'c': n_points * [3]}

    def test_1(self):
        assert_array_equal(rpg._reduce_header(self.header),
                           {'a': 1, 'b': 2, 'c': 3})

    def test_2(self):
        self.header['a'][50] = 10
        with pytest.raises(AssertionError):
            assert_array_equal(rpg._reduce_header(self.header),
                               {'a': 1, 'b': 2, 'c': 3})


def test_get_rpg_time():
    secs_in_day = 24 * 60 * 60
    assert rpg._get_rpg_time(0) == ['2001', '01', '01']
    assert rpg._get_rpg_time(secs_in_day - 1) == ['2001', '01', '01']
    assert rpg._get_rpg_time(secs_in_day) == ['2001', '01', '02']
