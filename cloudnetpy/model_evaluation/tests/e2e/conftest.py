import pytest

args = ["site", "date", "input", "output", "full_path"]


def pytest_addoption(parser) -> None:
    for arg in args:
        parser.addoption(f"--{arg}", action="store")


@pytest.fixture()
def params(request) -> dict:
    return {arg: request.config.getoption(f"--{arg}") for arg in args}
