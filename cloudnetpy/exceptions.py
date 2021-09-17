from typing import Optional


class InconsistentDataError(Exception):
    """Internal exception class."""
    def __init__(self, msg: Optional[str] = ''):
        self.message = msg
        super().__init__(self.message)
