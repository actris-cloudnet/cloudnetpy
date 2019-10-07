#!/home/korpinen/anaconda3/bin/python3
import sys
import os
sys.path.insert(0, os.path.split(os.path.dirname(os.path.abspath(__file__)))[0])
import pytest


def main(test_file):

    pytest.main(['-s',
                 'test_metadata.py',
                 '--test_file', test_file])


if __name__ == "__main__":
    main(*sys.argv[1:])
