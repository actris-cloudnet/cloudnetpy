import numpy as np
import pytest
from cloudnetpy.products.classification import _get_target_classification, _get_detection_status


class CategorizeBits:
    def __init__(self):
        self.category_bits = {'droplet': np.asarray([[1, 0, 1, 1, 1, 1],
                                                     [0, 1, 1, 1, 0, 0]], dtype=bool),
                              'falling': np.asarray([[0, 0, 0, 0, 1, 0],
                                                     [0, 0, 0, 1, 1, 1]], dtype=bool),
                              'cold': np.asarray([[0, 0, 1, 1, 0, 0],
                                                  [0, 1, 1, 1, 0, 1]], dtype=bool),
                              'melting': np.asarray([[1, 0, 1, 0, 0, 0],
                                                     [1, 1, 0, 0, 0, 0]], dtype=bool),
                              'aerosol': np.asarray([[1, 0, 1, 0, 0, 0],
                                                      [0, 0, 0, 0, 0, 0]], dtype=bool),
                              'insect': np.asarray([[1, 1, 0, 0, 0, 0],
                                                     [0, 0, 1, 0, 0, 0]], dtype=bool),
                               }
        self.quality_bits = {'radar': np.asarray([[0, 0, 0, 1, 1, 1],
                                                  [1, 0, 0, 1, 1, 1]], dtype=bool),
                             'lidar': np.asarray([[1, 1, 1, 1, 0, 0],
                                                  [1, 1, 0, 1, 1, 0]], dtype=bool),
                             'clutter': np.asarray([[0, 0, 1, 1, 0, 0],
                                                    [0, 0, 0, 0, 0, 0]], dtype=bool),
                             'molecular': np.asarray([[1, 0, 0, 1, 0, 0],
                                                      [0, 1, 0, 0, 0, 0]], dtype=bool),
                             'attenuated': np.asarray([[1, 1, 1, 0, 0, 1],
                                                       [0, 1, 1, 0, 0, 0]], dtype=bool),
                             'corrected': np.asarray([[1, 0, 0, 0, 0, 0],
                                                      [1, 1, 0, 0, 0, 0]], dtype=bool)}


@pytest.mark.parametrize('value', [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
def test_get_target_classification(value):
    bits = CategorizeBits()
    x = _get_target_classification(bits)
    assert value in x


@pytest.mark.parametrize('value', [
    0, 1, 2, 3, 4, 5, 6, 7])
def test_get_detection_status_value(value):
    bits = CategorizeBits()
    x = _get_detection_status(bits)
    assert value in x
