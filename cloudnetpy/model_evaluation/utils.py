import os


def file_exists(file_path: str):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0
