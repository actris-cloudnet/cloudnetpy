import numpy as np
from numpy.testing import assert_array_equal

from cloudnetpy.products import classification
from cloudnetpy.products.product_tools import CategoryBits, QualityBits


class CategorizeBits:
    def __init__(self):
        self.category_bits = CategoryBits(
            droplet=np.asarray([[1, 0, 1, 1, 1, 1], [0, 1, 1, 1, 0, 0]], dtype=bool),
            falling=np.asarray([[0, 0, 0, 0, 1, 0], [0, 0, 0, 1, 1, 1]], dtype=bool),
            freezing=np.asarray([[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 0, 1]], dtype=bool),
            melting=np.asarray([[1, 0, 1, 0, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=bool),
            aerosol=np.asarray([[1, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=bool),
            insect=np.asarray([[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=bool),
        )
        self.quality_bits = QualityBits(
            radar=np.asarray([[0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 1, 1]], dtype=bool),
            lidar=np.asarray([[1, 1, 1, 1, 0, 0], [1, 1, 0, 1, 1, 0]], dtype=bool),
            clutter=np.asarray([[0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0]], dtype=bool),
            molecular=np.asarray(
                [[1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 0]],
                dtype=bool,
            ),
            attenuated_liquid=np.asarray(
                [[1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 0, 0]],
                dtype=bool,
            ),
            corrected_liquid=np.asarray(
                [[1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]],
                dtype=bool,
            ),
            attenuated_rain=np.zeros((2, 6), dtype=bool),
            corrected_rain=np.zeros((2, 6), dtype=bool),
            attenuated_melting=np.zeros((2, 6), dtype=bool),
            corrected_melting=np.zeros((2, 6), dtype=bool),
        )


def test_get_target_classification():
    bits = CategorizeBits()
    target_classification = classification._get_target_classification(bits)  # type: ignore
    expected = np.array([[8, 9, 8, 0, 3, 1], [6, 7, 9, 5, 2, 4]])
    assert_array_equal(target_classification, expected)


def test_get_detection_status():
    bits = CategorizeBits()
    detection_status = classification._get_detection_status(bits)  # type: ignore
    expected = np.array([[9, 4, 8, 8, 5, 2], [7, 9, 4, 3, 3, 5]])
    assert_array_equal(detection_status, expected)


def test_find_cloud_mask():
    target_classification = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    cloud_mask = classification._find_cloud_mask(target_classification)
    expected = np.array([0, 1, 0, 1, 1, 1, 0, 0])
    assert_array_equal(cloud_mask, expected)
