import os
from os import PathLike


def file_exists(file_path: str | PathLike) -> bool:
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0
