from cloudnetpy import output
import pytest
import netCDF4
from cloudnetpy import utils


@pytest.mark.parametrize("short_id, result", [
    ('lwc', 'liquid water content'),
    ('iwc', 'ice water content'),
    ('drizzle', 'drizzle'),
    ('classification', 'classification'),
])
def test_get_identifier(short_id, result):
    assert output._get_identifier(short_id) == result


def test_get_identifier_raise():
    with pytest.raises(ValueError):
        output._get_identifier('dummy')


class RootGrp:
    def __init__(self):
        self.cloudnet_file_type = None
        self.history = None
        self.dataset = History()


class History:
    def __init__(self):
        self.history = None


def test_add_file_type():
    root_group = RootGrp()
    output.add_file_type(root_group, 'drizzle')
    assert root_group.cloudnet_file_type == 'drizzle'


@pytest.fixture
def fake_nc_file(tmpdir_factory):
    """Creates a simple categorize for testing."""
    file_name = tmpdir_factory.mktemp("data").join("nc_file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    root_grp.a = 1
    root_grp.b = 2
    root_grp.c = 3
    root_grp.file_uuid = 'abcde'
    var = root_grp.createVariable('a', 'f8')
    var[:] = 1.0
    var = root_grp.createVariable('b', 'f8')
    var[:] = 2.0
    var = root_grp.createVariable('c', 'i4')
    var[:] = 3
    root_grp.close()
    return file_name


def test_copy_global(tmpdir_factory, fake_nc_file):
    file = tmpdir_factory.mktemp("data").join("nc_file.nc")
    root_grp = netCDF4.Dataset(file, "w", format="NETCDF4_CLASSIC")
    attr_list = ('a', 'b', 'c')
    source = netCDF4.Dataset(fake_nc_file)
    output.copy_global(source, root_grp, attr_list)
    for attr in attr_list:
        assert getattr(root_grp, attr) == getattr(source, attr)
    root_grp.close()
    source.close()


def test_copy_variables(tmpdir_factory, fake_nc_file):
    file = tmpdir_factory.mktemp("data").join("nc_file.nc")
    root_grp = netCDF4.Dataset(file, "w", format="NETCDF4_CLASSIC")
    var_list = ('a', 'b', 'c')
    source = netCDF4.Dataset(fake_nc_file)
    output.copy_variables(source, root_grp, var_list)
    for var in var_list:
        assert source.variables[var][:] == root_grp.variables[var][:]
    root_grp.close()


def test_merge_history():
    root = RootGrp()
    file_type = 'dummy'
    source1 = RootGrp()
    source1.dataset.history = 'some history x'
    source2 = RootGrp()
    source2.dataset.history = 'some history y'
    output.merge_history(root, file_type, source1, source2)
    assert utils.is_timestamp(f"-{root.history[:19]}") is True
    assert root.history[19:] == ' - dummy file created\nsome history x\nsome history y'


def test_get_source_uuids():
    uuid1, uuid2 = 'simorules', 'abcdefg'
    source1, source2, source3 = RootGrp(), RootGrp(), RootGrp()
    source1.dataset.file_uuid = uuid1
    source2.dataset.file_uuid = uuid2
    res = output.get_source_uuids(source1, source2, source3)
    for value in (uuid1, uuid2, ', '):
        assert value in res


def test_get_old_uuid(fake_nc_file):
    assert output._get_old_uuid(True, fake_nc_file) == 'abcde'
    assert output._get_old_uuid(False, fake_nc_file) is None
    assert output._get_old_uuid(True, 'new_file.nc') is None
    assert output._get_old_uuid(False, 'new_file.nc') is None


def test_add_standard_global_attributes(tmpdir_factory, fake_nc_file):
    file = tmpdir_factory.mktemp("data").join("nc_file.nc")
    root_grp = netCDF4.Dataset(file, "w", format="NETCDF4_CLASSIC")
    output._add_standard_global_attributes(root_grp, 'abcd')
    assert root_grp.file_uuid == 'abcd'
    assert root_grp.Conventions == 'CF-1.7'
    output._add_standard_global_attributes(root_grp)
    assert root_grp.file_uuid != 'abcd'
    root_grp.close()
