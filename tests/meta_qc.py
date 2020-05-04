#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from tests import utils


def main(test_file, log_file):
    utils.init_logger(test_file, log_file)

    res = pytest.main(['-s',
                       '-v',
                       '--tb=line',
                       'meta/test_metadata.py',
                       '--test_file', test_file])
    if res.name != 'OK':
        sys.exit(1)


if __name__ == "__main__":
    main(*sys.argv[1:])
