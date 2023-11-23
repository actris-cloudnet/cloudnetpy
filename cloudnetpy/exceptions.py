class CloudnetException(Exception):
    """Base class for exceptions in this module."""


class InconsistentDataError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str):
        super().__init__(msg)


class DisdrometerDataError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str):
        super().__init__(msg)


class RadarDataError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str):
        super().__init__(msg)


class WeatherStationDataError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str = "Unable to read the file"):
        super().__init__(msg)


class ModelDataError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str = "Invalid model file: not enough proper profiles"):
        super().__init__(msg)


class ValidTimeStampError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str = "No valid timestamps found"):
        super().__init__(msg)


class MissingInputFileError(CloudnetException):
    """Internal exception class."""

    def __init__(self, msg: str = "Missing required input files"):
        super().__init__(msg)
