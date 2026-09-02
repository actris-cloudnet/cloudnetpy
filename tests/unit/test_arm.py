import datetime
from os import path
from pathlib import Path

import netCDF4
import pytest

from cloudnetpy import arm

SCRIPT_PATH = path.dirname(path.realpath(__file__))
DATA_PATH = Path(SCRIPT_PATH) / "data"
DATE = datetime.date(2010, 3, 10)
RAW_FILES = {
    "radar": [DATA_PATH / "mmcr/sgpmmcrmomC1.b1.20100310.000047.cdf"],
    "mwr": [DATA_PATH / "mwrlos/sgpmwrlosC1.b1.20100310.000025.cdf"],
}
LIDAR_FILES = [DATA_PATH / "arm-ceilo/sgpceilC1.b1.20220601.000000.nc"]
SITE_META = {"name": "Southern Great Plains", "altitude": 316}


class FakeResponse:
    def __init__(
        self, json_data=None, content=b"", text="", headers=None, status_code=200
    ):
        self._json = json_data
        self._content = content
        self.text = text
        self.headers = headers or {}
        self.status_code = status_code
        self.ok = status_code < 400

    def close(self):
        pass

    def raise_for_status(self):
        pass

    def json(self):
        if self._json is None:
            raise arm.requests.JSONDecodeError("bad", "", 0)
        return self._json

    def iter_content(self, chunk_size):
        yield self._content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


@pytest.fixture
def credentials(monkeypatch):
    monkeypatch.setenv(arm.ARM_USER_ENV, "user")
    monkeypatch.setenv(arm.ARM_TOKEN_ENV, "token")


def test_datastream():
    streams = [arm.get_datastream("arm-sgp", i) for i in arm.ARM_INSTRUMENTS]
    assert streams == [
        "sgpkazrcfrgeC1.a1",
        "sgpkazr2cfrgeC1.a1",
        "sgpkazrcorgeC1.c1",
        "sgpkazrgeC1.a1",
        "sgpmmcrmomC1.b1",
        "sgpceilC1.b1",
        "sgpmwrlosC1.b1",
        "sgpldC1.b1",
    ]
    assert arm.get_datastream("arm-darwin", arm.ARM_INSTRUMENTS[4]) == (
        "twpmmcrmomC3.b1"
    )
    assert arm.get_datastream("arm-andoya", arm.ARM_INSTRUMENTS[0]) == (
        "anxkazrcfrgeM1.a1"
    )


def test_unknown_site():
    with pytest.raises(arm.ArmDataError, match="Unknown ARM site"):
        arm.get_datastream("hyytiala", arm.ARM_INSTRUMENTS[0])


def test_missing_credentials(monkeypatch):
    monkeypatch.delenv(arm.ARM_USER_ENV, raising=False)
    monkeypatch.delenv(arm.ARM_TOKEN_ENV, raising=False)
    with pytest.raises(arm.ArmDataError, match="credentials"):
        arm.get_credentials()


def test_credentials(credentials):
    assert arm.get_credentials() == "user:token"


def test_query_files(monkeypatch, credentials):
    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params))
        return FakeResponse({"files": ["b.cdf", "a.cdf"], "status": "success"})

    monkeypatch.setattr(arm.requests, "get", fake_get)
    files = arm.query_files("sgpmmcrmomC1.b1", DATE)
    assert files == ["a.cdf", "b.cdf"]
    url, params = calls[0]
    assert url.endswith("/query")
    assert params["user"] == "user:token"
    assert params["ds"] == "sgpmmcrmomC1.b1"
    assert params["start"] == params["end"] == "2010-03-10"


def test_query_failure(monkeypatch, credentials):
    monkeypatch.setattr(
        arm.requests, "get", lambda *a, **k: FakeResponse({"status": "error"})
    )
    with pytest.raises(arm.ArmDataError, match="query failed"):
        arm.query_files("sgpmmcrmomC1.b1", DATE)
    monkeypatch.setattr(
        arm.requests, "get", lambda *a, **k: FakeResponse(status_code=500)
    )
    with pytest.raises(arm.ArmDataError, match="HTTP 500"):
        arm.query_files("sgpmmcrmomC1.b1", DATE)


def test_query_invalid_response(monkeypatch, credentials):
    monkeypatch.setattr(
        arm.requests, "get", lambda *a, **k: FakeResponse(text="Invalid username.")
    )
    with pytest.raises(arm.ArmDataError, match="Invalid response"):
        arm.query_files("sgpmmcrmomC1.b1", DATE)


def test_download_file(monkeypatch, credentials, tmp_path):
    calls = []

    def fake_get(url, params, headers, timeout, stream):
        if headers.get("Range") == "bytes=0-0":
            return FakeResponse(
                headers={"Content-Range": "bytes 0-0/4"}, status_code=206
            )
        calls.append(params)
        return FakeResponse(content=b"data", headers={"Content-Length": "4"})

    monkeypatch.setattr(arm.requests, "get", fake_get)
    filepath = arm.download_file("x.cdf", tmp_path)
    assert filepath == tmp_path / "x.cdf"
    assert filepath.read_bytes() == b"data"
    assert calls[0] == {"user": "user:token", "file": "x.cdf"}
    # Existing file is not re-downloaded unless forced
    arm.download_file("x.cdf", tmp_path)
    assert len(calls) == 1
    arm.download_file("x.cdf", tmp_path, force=True)
    assert len(calls) == 2


def test_download_resumes_truncated_transfer(monkeypatch, credentials, tmp_path):
    data = b"0123456789"
    requests_made = []

    def fake_get(url, params, headers, timeout, stream):
        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers else 0
        requests_made.append(start)
        chunk = data[start : start + 4]  # Server drops the connection after 4 bytes
        return FakeResponse(
            content=chunk,
            headers={
                "Content-Length": str(len(data) - start),
                **(
                    {"Content-Range": f"bytes {start}-{len(data) - 1}/{len(data)}"}
                    if start
                    else {}
                ),
            },
            status_code=206 if start else 200,
        )

    monkeypatch.setattr(arm.requests, "get", fake_get)
    filepath = arm.download_file("x.cdf", tmp_path)
    assert filepath.read_bytes() == data
    assert requests_made == [0, 4, 8]
    assert not (tmp_path / "x.cdf.part").exists()


def test_download_resumes_truncated_existing_file(monkeypatch, credentials, tmp_path):
    data = b"0123456789"
    (tmp_path / "x.cdf").write_bytes(data[:6])  # Left over from an old version

    def fake_get(url, params, headers, timeout, stream):
        if headers.get("Range") == "bytes=0-0":
            return FakeResponse(
                headers={"Content-Range": f"bytes 0-0/{len(data)}"}, status_code=206
            )
        start = int(headers["Range"].split("=")[1].rstrip("-")) if headers else 0
        return FakeResponse(
            content=data[start:],
            headers={"Content-Range": f"bytes {start}-9/{len(data)}"},
            status_code=206,
        )

    monkeypatch.setattr(arm.requests, "get", fake_get)
    assert arm.download_file("x.cdf", tmp_path).read_bytes() == data


def test_download_chunked_encoding_error_retried(monkeypatch, credentials, tmp_path):
    attempts = []

    def fake_get(url, params, headers, timeout, stream):
        attempts.append(1)
        if len(attempts) == 1:
            raise arm.requests.exceptions.ChunkedEncodingError("broken")
        return FakeResponse(content=b"data", headers={"Content-Length": "4"})

    monkeypatch.setattr(arm.requests, "get", fake_get)
    assert arm.download_file("x.cdf", tmp_path).read_bytes() == b"data"


def test_download_finishes_complete_part_file(monkeypatch, credentials, tmp_path):
    data = b"0123456789"
    (tmp_path / "x.cdf.part").write_bytes(data)  # Killed before rename
    requests_made = []

    def fake_get(url, params, headers, timeout, stream):
        requests_made.append(headers.get("Range"))
        return FakeResponse(
            headers={"Content-Range": f"bytes 0-0/{len(data)}"}, status_code=206
        )

    monkeypatch.setattr(arm.requests, "get", fake_get)
    assert arm.download_file("x.cdf", tmp_path).read_bytes() == data
    assert requests_made == ["bytes=0-0"]
    assert not (tmp_path / "x.cdf.part").exists()


def test_download_restarts_on_unsatisfiable_range(monkeypatch, credentials, tmp_path):
    data = b"0123456789"
    (tmp_path / "x.cdf.part").write_bytes(data)
    requests_made = []

    class ErrorResponse(FakeResponse):
        def raise_for_status(self):
            raise arm.requests.HTTPError(response=self)  # type: ignore[arg-type]

    def fake_get(url, params, headers, timeout, stream):
        requests_made.append(headers.get("Range"))
        if headers.get("Range") == "bytes=0-0":
            return FakeResponse(status_code=200)  # Size unknown
        if headers:
            return ErrorResponse(status_code=416)
        return FakeResponse(content=data, headers={"Content-Length": str(len(data))})

    monkeypatch.setattr(arm.requests, "get", fake_get)
    assert arm.download_file("x.cdf", tmp_path).read_bytes() == data
    assert requests_made == ["bytes=0-0", f"bytes={len(data)}-", None]


def test_download_not_found(monkeypatch, credentials, tmp_path):
    class ErrorResponse(FakeResponse):
        def raise_for_status(self):
            raise arm.requests.HTTPError(response=self)  # type: ignore[arg-type]

    monkeypatch.setattr(
        arm.requests, "get", lambda *a, **k: ErrorResponse(status_code=404)
    )
    with pytest.raises(arm.ArmFileNotFoundError, match="x.cdf not found"):
        arm.download_file("x.cdf", tmp_path)


def test_download_gives_up(monkeypatch, credentials, tmp_path):
    monkeypatch.setattr(
        arm.requests,
        "get",
        lambda *a, **k: FakeResponse(content=b"", headers={"Content-Length": "9"}),
    )
    with pytest.raises(arm.ArmDataError, match="Incomplete"):
        arm.download_file("x.cdf", tmp_path, max_attempts=3)
    assert not (tmp_path / "x.cdf").exists()


def test_download_connection_error_retried(monkeypatch, credentials, tmp_path):
    attempts = []

    def fake_get(url, params, headers, timeout, stream):
        attempts.append(1)
        if len(attempts) < 3:
            raise arm.requests.ConnectionError("reset")
        return FakeResponse(content=b"data", headers={"Content-Length": "4"})

    monkeypatch.setattr(arm.requests, "get", fake_get)
    assert arm.download_file("x.cdf", tmp_path).read_bytes() == b"data"
    assert len(attempts) == 3


def test_fetch_files(monkeypatch, credentials, tmp_path):
    available = {"sgpmmcrmomC1.b1", "sgpceilC1.b1"}

    def fake_query(datastream, date, credentials):
        return [f"{datastream}.20100310.cdf"] if datastream in available else []

    def fake_download(filename, output_dir, credentials, *, force):
        return Path(output_dir) / filename

    monkeypatch.setattr(arm, "query_files", fake_query)
    monkeypatch.setattr(arm, "download_file", fake_download)
    files = arm.fetch_files("arm-sgp", DATE, tmp_path)
    assert set(files) == {"radar", "lidar"}
    assert files["radar"] == [
        tmp_path / "sgpmmcrmomC1.b1" / "sgpmmcrmomC1.b1.20100310.cdf"
    ]
    assert (tmp_path / "sgpceilC1.b1").is_dir()
    files = arm.fetch_files("arm-sgp", DATE, tmp_path, products=("lidar",))
    assert set(files) == {"lidar"}


def test_fetch_files_prefers_newer_datastream(monkeypatch, credentials, tmp_path):
    def fake_query(datastream, date, credentials):
        return [f"{datastream}.20220601.000007.nc", f"{datastream}.20220601.010009.nc"]

    def fake_download(filename, output_dir, credentials, *, force):
        return Path(output_dir) / filename

    monkeypatch.setattr(arm, "query_files", fake_query)
    monkeypatch.setattr(arm, "download_file", fake_download)
    files = arm.fetch_files("arm-sgp", datetime.date(2022, 6, 1), tmp_path)
    assert len(files["radar"]) == 2
    assert files["radar"][0].parent.name == "sgpkazrcfrgeC1.a1"
    assert files["lidar"][0].parent.name == "sgpceilC1.b1"


def test_fetch_files_skips_unavailable_datastream(monkeypatch, credentials, tmp_path):
    def fake_query(datastream, date, credentials):
        return [f"{datastream}.20120124.000000.nc"]

    def fake_download(filename, output_dir, credentials, *, force):
        if filename.startswith("gankazrcorgeM1.c1"):
            raise arm.ArmFileNotFoundError(filename)
        return Path(output_dir) / filename

    monkeypatch.setattr(arm, "query_files", fake_query)
    monkeypatch.setattr(arm, "download_file", fake_download)
    files = arm.fetch_files("arm-maldives", datetime.date(2012, 1, 24), tmp_path)
    assert files["radar"][0].parent.name == "gankazrcfrgeM1.a1"


def test_convert_to_l1b(tmp_path):
    l1b = arm.convert_to_l1b("arm-sgp", DATE, RAW_FILES, tmp_path, SITE_META)
    assert set(l1b) == {"radar", "mwr"}
    assert l1b["radar"] == str(tmp_path / "20100310_arm-sgp_mmcr.nc")
    assert l1b["mwr"] == str(tmp_path / "20100310_arm-sgp_wvr-1100.nc")
    for product, file_type in (("radar", "radar"), ("mwr", "mwr")):
        with netCDF4.Dataset(l1b[product]) as nc:
            assert nc.cloudnet_file_type == file_type
            assert nc.location == SITE_META["name"]


def test_convert_to_l1b_lidar_calibration(tmp_path):
    date = datetime.date(2022, 6, 1)
    l1b = arm.convert_to_l1b(
        "arm-sgp", date, {"lidar": LIDAR_FILES}, tmp_path, SITE_META
    )
    assert l1b == {"lidar": str(tmp_path / "20220601_arm-sgp_cl31.nc")}
    with netCDF4.Dataset(l1b["lidar"]) as nc:
        assert nc.cloudnet_file_type == "lidar"
        assert nc.variables["calibration_factor"][:] == 1
    calibration = {"lidar": {"calibration_factor": 1.5}}
    l1b = arm.convert_to_l1b(
        "arm-sgp", date, {"lidar": LIDAR_FILES}, tmp_path, SITE_META, calibration
    )
    with netCDF4.Dataset(l1b["lidar"]) as nc:
        assert nc.variables["calibration_factor"][:] == 1.5


def test_find_l1b_files(tmp_path):
    date = datetime.date(2022, 6, 1)
    assert arm.find_l1b_files("arm-sgp", date, tmp_path) == {}
    (tmp_path / "20220601_arm-sgp_ct25k.nc").touch()
    (tmp_path / "20220601_arm-sgp_mmcr.nc").touch()
    assert arm.find_l1b_files("arm-sgp", date, tmp_path) == {
        "lidar": str(tmp_path / "20220601_arm-sgp_ct25k.nc"),
        "radar": str(tmp_path / "20220601_arm-sgp_mmcr.nc"),
    }


def test_convert_to_l1b_kazr(tmp_path):
    files = sorted((DATA_PATH / "kazr").glob("*.nc"))
    date = datetime.date(2022, 6, 1)
    l1b = arm.convert_to_l1b("arm-sgp", date, {"radar": files}, tmp_path, SITE_META)
    assert l1b == {"radar": str(tmp_path / "20220601_arm-sgp_kazr.nc")}


def test_convert_to_l1b_single_file_reader_with_many_files(tmp_path, caplog):
    date = datetime.date(2022, 6, 1)
    files = {"lidar": [LIDAR_FILES[0], LIDAR_FILES[0]]}
    l1b = arm.convert_to_l1b("arm-sgp", date, files, tmp_path, SITE_META)
    assert l1b == {"lidar": str(tmp_path / "20220601_arm-sgp_cl31.nc")}
    assert "Found 2 lidar files" in caplog.text


def test_convert_to_l1b_reader_error(monkeypatch, tmp_path, caplog):
    def broken_reader(*args, **kwargs):
        msg = "Radar mode not found"
        raise ValueError(msg)

    instrument = arm.ARM_INSTRUMENTS[4]._replace(reader=broken_reader)
    monkeypatch.setattr(arm, "ARM_INSTRUMENTS", (instrument, *arm.ARM_INSTRUMENTS[5:]))
    l1b = arm.convert_to_l1b("arm-sgp", DATE, RAW_FILES, tmp_path, SITE_META)
    assert set(l1b) == {"mwr"}
    assert "Failed to process radar: Radar mode not found" in caplog.text


def test_unknown_file(tmp_path):
    with pytest.raises(arm.ArmDataError, match="Unknown ARM"):
        arm._find_instrument("arm-sgp", "radar", Path("foo.nc"))


def test_convert_to_l1b_wrong_date(tmp_path):
    date = datetime.date(2010, 3, 11)
    l1b = arm.convert_to_l1b("arm-sgp", date, RAW_FILES, tmp_path, SITE_META)
    assert l1b == {}
