class InconsistentDataError(Exception):
    """Internal exception class."""

    def __init__(self, msg: str = ""):
        self.message = msg
        super().__init__(self.message)


class DisdrometerDataError(Exception):
    """Internal exception class."""

    def __init__(self, msg: str = ""):
        self.message = msg
        super().__init__(self.message)


class RadarDataError(Exception):
    """Internal exception class."""

    def __init__(self, msg: str = ""):
        self.message = msg
        super().__init__(self.message)


class WeatherStationDataError(Exception):
    """Internal exception class."""

    def __init__(self, msg: str = ""):
        self.message = msg
        super().__init__(self.message)


class ModelDataError(Exception):
    """Internal exception class."""

    def __init__(self, msg: str = "Invalid model file: not enough proper profiles"):
        self.message = msg
        super().__init__(self.message)


class ValidTimeStampError(Exception):
    """Internal exception class."""

    def __init__(self, msg: str = "No valid timestamps found"):
        self.message = msg
        super().__init__(self.message)


class MissingInputFileError(Exception):
    """Internal exception class."""

    def __init__(self, msg: str = "Missing required input files"):
        self.message = msg
        super().__init__(self.message)
