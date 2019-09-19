import os
import subprocess


def get_test_path():
    return os.path.dirname(os.path.abspath(__file__))


def check_quality(file, log_file):
    """Runs data quality tests for the given file."""
    test_path = get_test_path()
    script = f"{test_path}/quality_control.py"
    subprocess.call([script, file, log_file])

