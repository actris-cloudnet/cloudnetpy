import netCDF4
import pytest

from cloudnetpy import output, utils
from cloudnetpy.metadata import MetaData


@pytest.mark.parametrize(
    "short_id, result",
    [
        ("lwc", "liquid water content"),
        ("iwc", "ice water content"),
        ("drizzle", "drizzle"),
        ("classification", "classification"),
    ],
)
def test_get_identifier(short_id, result):
    assert output._get_identifier(short_id) == result


def test_get_identifier_raise():
    with pytest.raises(ValueError):
        output._get_identifier("dummy")


class RootGrp:
    def __init__(self):
        self.cloudnet_file_type = None
        self.history = None
        self.dataset = History()


class History:
    def __init__(self):
        self.history = None


@pytest.fixture
def fake_nc_file(tmpdir_factory):
    """Creates a simple categorize for testing."""
    file_name = tmpdir_factory.mktemp("data").join("nc_file.nc")
    with netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC") as root_grp:
        root_grp.a = 1
        root_grp.b = 2
        root_grp.c = 3
        root_grp.file_uuid = "abcde"
        var = root_grp.createVariable("a", "f8")
        var[:] = 1.0
        var = root_grp.createVariable("b", "f8")
        var[:] = 2.0
        var = root_grp.createVariable("c", "i4")
        var[:] = 3
    return file_name


def test_copy_global(tmpdir_factory, fake_nc_file):
    file = tmpdir_factory.mktemp("data").join("nc_file.nc")
    with netCDF4.Dataset(file, "w", format="NETCDF4_CLASSIC") as root_grp:
        attr_list = ("a", "b", "c")
        with netCDF4.Dataset(fake_nc_file) as source:
            output.copy_global(source, root_grp, attr_list)
            for attr in attr_list:
                assert getattr(root_grp, attr) == getattr(source, attr)


def test_copy_variables(tmpdir_factory, fake_nc_file):
    file = tmpdir_factory.mktemp("data").join("nc_file.nc")
    with netCDF4.Dataset(file, "w", format="NETCDF4_CLASSIC") as root_grp:
        var_list = ("a", "b", "c")
        with netCDF4.Dataset(fake_nc_file) as source:
            output.copy_variables(source, root_grp, var_list)
            for var in var_list:
                assert source.variables[var][:] == root_grp.variables[var][:]


def test_merge_history():
    root = RootGrp()
    file_type = "dummy"
    source1 = RootGrp()
    source1.dataset.history = "20:00 some history x"
    source2 = RootGrp()
    source2.dataset.history = "21:00 some history y"
    output.merge_history(root, file_type, {"a": source1, "b": source2})
    history = str(root.history)
    assert utils.is_timestamp(f"-{history[:19]}") is True
    assert (
        history[19:] == " +00:00 - dummy file created\n21:00 some history y\n20:00 some history x"
    )


def test_get_source_uuids():
    uuid1, uuid2 = "simorules", "abcdefg"
    source1, source2, source3, source4 = RootGrp(), RootGrp(), RootGrp(), RootGrp()
    source1.dataset.file_uuid = uuid1  # type: ignore
    source2.dataset.file_uuid = uuid2  # type: ignore
    source3.dataset.file_uuid = uuid2  # type: ignore
    res = output.get_source_uuids(source1, source2, source3, source4)
    for value in (uuid1, uuid2, ", "):
        assert value in res
    assert len(res) == len(uuid1) + len(uuid2) + 2


def test_add_standard_global_attributes(tmpdir_factory):
    file = tmpdir_factory.mktemp("data").join("nc_file.nc")
    with netCDF4.Dataset(file, "w", format="NETCDF4_CLASSIC") as root_grp:
        output._add_standard_global_attributes(root_grp, "abcd")
        assert root_grp.file_uuid == "abcd"
        assert root_grp.Conventions == "CF-1.8"
        output._add_standard_global_attributes(root_grp)
        assert root_grp.file_uuid != "abcd"


def test_add_time_attribute():
    attr = MetaData(long_name="Some name", units="xy")
    attributes = {"kissa": attr}
    date = ["2020", "01", "12"]
    new_attributes = output.add_time_attribute(attributes, date)
    assert new_attributes["time"].units == "hours since 2020-01-12 00:00:00 +00:00"
    assert new_attributes["kissa"].units == "xy"


def test_fix_attribute_name(tmpdir_factory):
    file = tmpdir_factory.mktemp("data").join("nc_file.nc")
    nc = netCDF4.Dataset(file, "w", format="NETCDF4_CLASSIC")
    var = nc.createVariable("a", "f8")
    var[:] = 1.0
    var.unit = "m"
    output.fix_attribute_name(nc)
    assert hasattr(nc.variables["a"], "units") is True
    assert hasattr(nc.variables["a"], "unit") is False
    assert nc.variables["a"].units == "m"
