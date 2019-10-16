#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
import pytest
from tests.utils import init_logger


def main(test_file, log_file):
    init_logger(test_file, log_file)

    pytest.main(['-s',
                 '-v',
                 '--tb=line',
                 'data_quality/test_data.py',
                 '--test_file', test_file])


if __name__ == "__main__":
    main(*sys.argv[1:])
