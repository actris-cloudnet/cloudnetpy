from tests import utils
import subprocess
import pytest


def check_metadata(file, log_file=None):
    """Runs metadata checks for the given file.

    Args:
        file (str): Name of the file to be tested.
        log_file (str): Name of the log file.

    """
    test_path = utils.get_test_path()
    script = f"{test_path}/meta_qc.py"
    subprocess.call([script, file])


def check_data_quality(file, log_file=None):
    """Runs data quality checks for the given file.

    Args:
        file (str): Name of the file to be tested.
        log_file (str): Name of the log file.

    """
    test_path = utils.get_test_path()
    script = f"{test_path}/data_qc.py"
    subprocess.call([script, file])


def run_unit_tests():
    """Runs all CloudnetPy unit tests."""
    pytest.main(["-s", "unit/"])
