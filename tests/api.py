from tests import utils
import subprocess
import pytest


def check_metadata(file):
    """Runs metadata checks for the given file."""
    test_path = utils.get_test_path()
    script = f"{test_path}/meta_qc.py"
    subprocess.call([script, file])


def run_unit_tests():
    pytest.main(["-s", "unit/"])
