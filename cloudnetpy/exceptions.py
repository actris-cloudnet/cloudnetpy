class CloudnetException(Exception):
    """Base class for exceptions in this module."""


class InconsistentDataError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class DisdrometerDataError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class RadarDataError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class LidarDataError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class PlottingError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)


class ModelDataError(CloudnetException):
    """Internal exception class."""

    def __init__(
        self, msg: str = "Invalid model file: not enough proper profiles"
    ) -> None:
        super().__init__(msg)


class ValidTimeStampError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str = "No valid timestamps found") -> None:
        super().__init__(msg)


class HatproDataError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str = "Invalid HATPRO file") -> None:
        super().__init__(msg)


class InvalidSourceFileError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str = "Invalid source file") -> None:
        super().__init__(msg)
