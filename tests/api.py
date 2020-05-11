import subprocess
from uuid import UUID
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
    print(f"Checking metadata of {file}")
    try:
        subprocess.check_call([script, file, _validate_log_file(log_file)])
    except subprocess.CalledProcessError:
        raise


def check_data_quality(file, log_file=None):
    """Runs data quality checks for the given file.

    Args:
        file (str): Name of the file to be tested.
        log_file (str, optional): Name of the log file where warning
            messages are stored.

    """
    test_path = utils.get_test_path()
    script = f"{test_path}/data_qc.py"
    print(f"Checking data quality of {file}")
    try:
        subprocess.check_call([script, file, _validate_log_file(log_file)])
    except subprocess.CalledProcessError:
        raise


def check_is_valid_uuid(uuid):
    try:
        UUID(uuid, version=4)
    except (ValueError, TypeError):
        raise AssertionError(f'{uuid} is not a valid UUID.')


def _validate_log_file(log_file):
    return log_file if log_file else ''


def run_unit_tests():
    """Runs all CloudnetPy unit tests."""
    pytest.main(["-s", "unit/"])
