""" This module contains unit tests for atmos-module. """
import sys
sys.path.append('../cloudnetpy')
import numpy as np
import numpy.ma as ma
from numpy.testing import assert_array_almost_equal
import pytest
from cloudnetpy import categorize

