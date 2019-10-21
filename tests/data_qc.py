#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from tests import utils


def main(test_file, log_file):
    utils.init_logger(test_file, log_file)

    pytest.main(['-s',
                 '-v',
                 '--tb=line',
                 'data_quality/test_data.py',
                 '--test_file', test_file])


if __name__ == "__main__":
    main(*sys.argv[1:])
