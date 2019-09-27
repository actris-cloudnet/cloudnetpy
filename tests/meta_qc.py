#!/usr/bin/python3
import sys
import os
sys.path.insert(0, os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
import pytest
from tests import utils


def main(test_file):

    marker = utils.get_file_type(test_file)

    if marker in ('iwc', 'lwc', 'drizzle', 'classification'):
        marker = f"{marker} or product"

    pytest.main(['-s',
                 'meta/',
                 '--test_file', test_file,
                 '-m', marker])


if __name__ == "__main__":
    main(*sys.argv[1:])
