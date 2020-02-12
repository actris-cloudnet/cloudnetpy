import subprocess
import pytest
from tests import utils


def check_metadata(file, log_file=None):
    """Runs metadata checks for the given file.

    Args:
        file (str): Name of the file to be tested.
        log_file (str, optional): Name of the log file where warning
            messages are stored.

    """
    test_path = utils.get_test_path()
    script = f"{test_path}/meta_qc.py"
    subprocess.call([script, file, _validate_log_file(log_file)])


def check_data_quality(file, log_file=None):
    """Runs data quality checks for the given file.

    Args:
        file (str): Name of the file to be tested.
        log_file (str, optional): Name of the log file where warning
            messages are stored.

    """
    test_path = utils.get_test_path()
    script = f"{test_path}/data_qc.py"
    subprocess.call([script, file, _validate_log_file(log_file)])


def _validate_log_file(log_file):
    return log_file if log_file else ''


def run_unit_tests():
    """Runs all CloudnetPy unit tests."""
    pytest.main(["-s", "unit/"])
