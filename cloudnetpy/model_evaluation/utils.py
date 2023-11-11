import os
from pathlib import Path


def file_exists(file_path: str):
    return Path.is_file(Path(file_path)) and os.path.getsize(file_path) > 0
