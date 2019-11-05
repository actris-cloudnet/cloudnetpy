import numpy as np
import pytest
import netCDF4
from cloudnetpy.products.iwc import IwcSource


DIMENSIONS = ('time', 'height', 'model_time', 'model_height')
TEST_ARRAY = np.arange(5)  # Simple array for all dimension and variables


@pytest.fixture(scope='session')
def iwcsource_file(tmpdir_factory, file_metadata):
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    _create_dimensions(root_grp)
    _create_dimension_variables(root_grp)
    var = root_grp.createVariable('altitude', 'f8')
    var[:] = 1
    var.units = 'km'

    var = root_grp.createVariable('radar_frequency', 'f8')
    var[:] = 35.5 # Miten testata useampi vaihtoehto, 30-40 tai suurempaa
    var = root_grp.createVariable('temperature', 'f8', ('time', 'height'))
    var[:] = np.array([[280, 290, 278, 293, 276],
                       [280, 290, 278, 293, 276],
                      [280, 290, 278, 293, 276],
                       [280, 290, 278, 293, 276],
                       [280, 290, 278, 293, 276]])
    root_grp.close()
    return file_name


def _create_dimensions(root_grp):
    n_dim = len(TEST_ARRAY)
    for dim_name in DIMENSIONS:
        root_grp.createDimension(dim_name, n_dim)


def _create_dimension_variables(root_grp):
    for dim_name in DIMENSIONS:
        x = root_grp.createVariable(dim_name, 'f8', (dim_name,))
        x[:] = TEST_ARRAY
        if dim_name == 'height':
            x.units = 'm'


def test_iwc_wl_band(iwcsource_file):
    obj = IwcSource(iwcsource_file)
    compare = 0
    assert compare == obj.wl_band


def test_iwc_z_factor(iwcsource_file):
    obj = IwcSource(iwcsource_file)
    assert True


def test_iwc_spec_liq_atten(iwcsource_file):
    obj = IwcSource(iwcsource_file)
    compare = 1
    assert compare == obj.spec_liq_atten


def test_iwc_coeffs(iwcsource_file):
    obj = IwcSource(iwcsource_file)
    assert True


def test_iwc_temperature(iwcsource_file):
    obj = IwcSource(iwcsource_file)
    assert True


def test_iwc_mean_temperature(iwcsource_file):
    obj = IwcSource(iwcsource_file)
    assert True
