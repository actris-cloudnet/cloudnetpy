import numpy as np
import numpy.ma as ma
from cloudnetpy.plotting import meta_for_old_files
from numpy.testing import assert_array_equal


data_orig = ma.array([[0, 1, 2],
                      [3, 4, 5],
                      [6, 7, 8]])


def test_fix_old_data():
    data, name = meta_for_old_files.fix_old_data(data_orig, 'specific_humidity')
    assert_array_equal(data_orig, data)
    assert_array_equal(name, 'q')


def test_fix_old_data_2():
    data, name = meta_for_old_files.fix_old_data(data_orig, 'detection_status')
    assert_array_equal(data_orig, data)
    assert ma.count(data) == 7

