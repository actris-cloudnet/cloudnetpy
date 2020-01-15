import numpy as np
from collections import namedtuple
from numpy import testing
import pytest
import netCDF4
from cloudnetpy.products.iwc import IwcSource, _IceClassification

DIMENSIONS = ('time', 'height', 'model_time', 'model_height')
TEST_ARRAY = np.arange(3)


@pytest.fixture(scope='session')
def iwc_source_file(tmpdir_factory, file_metadata):
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    _create_dimensions(root_grp)
    _create_dimension_variables(root_grp)
    var = root_grp.createVariable('altitude', 'f8')
    var[:] = 1
    var.units = 'km'
    var = root_grp.createVariable('Z', 'f8', 'time')
    var[:] = [5, 10, 15]
    var = root_grp.createVariable('Z_sensitivity', 'f8', 'time')
    var[:] = [5, 10, 15]
    var = root_grp.createVariable('Z_bias', 'f8', 'time')
    var[:] = [5, 10, 15]
    var = root_grp.createVariable('radar_frequency', 'f8')
    var[:] = 35.5  # TODO: How to check with multiple options
    var = root_grp.createVariable('temperature', 'f8', ('time', 'height'))
    var[:] = np.array([[282, 280, 278],
                       [286, 284, 282],
                      [284, 282, 280]])
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


def test_iwc_wl_band(iwc_source_file):
    obj = IwcSource(iwc_source_file)
    compare = 0
    assert compare == obj.wl_band


def test_iwc_spec_liq_atten(iwc_source_file):
    obj = IwcSource(iwc_source_file)
    compare = 1
    assert compare == obj.spec_liq_atten


def test_iwc_z_factor(iwc_source_file):
    obj = IwcSource(iwc_source_file)
    Coefficients = namedtuple('Coefficients', 'K2liquid0')
    obj.coeffs = Coefficients(10)
    compare = -0.25
    assert compare == round(obj.z_factor, 3)


@pytest.mark.parametrize("result", [
    'K2liquid0', 'ZT', 'T', 'Z', 'c'])
def test_iwc_coeffs(result, iwc_source_file):
    obj = IwcSource(iwc_source_file)
    assert result in obj.coeffs._fields


def test_iwc_temperature(iwc_source_file):
    obj = IwcSource(iwc_source_file)
    compare = np.array([[8.85, 6.85, 4.85],
                       [12.85, 10.85, 8.85],
                       [10.85, 8.85, 6.85]])
    testing.assert_almost_equal(compare, obj.temperature)


def test_iwc_mean_temperature(iwc_source_file):
    obj = IwcSource(iwc_source_file)
    compare = np.array([10.85, 8.85, 6.85])
    testing.assert_almost_equal(compare, obj.mean_temperature)


@pytest.fixture(scope='session')
def iwc_cat_file(tmpdir_factory, file_metadata):
    file_name = tmpdir_factory.mktemp("data").join("file.nc")
    root_grp = netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC")
    n_points = 7
    root_grp.createDimension('time', n_points)
    var = root_grp.createVariable('time', 'f8', 'time')
    var[:] = np.arange(n_points)
    var = root_grp.createVariable('category_bits', 'i4', 'time')
    var[:] = [0, 1, 2, 4, 8, 16, 32]
    var = root_grp.createVariable('quality_bits', 'i4', 'time')
    var[:] = [0, 1, 2, 4, 8, 16, 32]
    var = root_grp.createVariable('is_rain', 'i4', 'time')
    var[:] = [0, 1, 1, 1, 0, 0, 0]
    var = root_grp.createVariable('is_undetected_melting', 'i4', 'time')
    var[:] = [0, 1, 0, 1, 0, 0, 1]
    root_grp.close()
    return file_name


@pytest.mark.parametrize("falling, cold, melting, insect, result", [
    (np.array([1, 1, 1, 1, 1, 1, 1]), np.array([0, 1, 0, 1, 0, 1, 0]),
     np.array([1, 1, 1, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0]),
     np.array([0, 0, 0, 1, 0, 1, 0]))])
def test_find_ice(falling, cold, melting,  insect, result, iwc_cat_file):
    obj = _IceClassification(iwc_cat_file)
    obj.category_bits['falling'] = falling
    obj.category_bits['cold'] = cold
    obj.category_bits['melting'] = melting
    obj.category_bits['insect'] = insect
    testing.assert_array_equal(obj._find_ice().data, result)


@pytest.mark.parametrize("falling, cold, melting, insect, result", [
    (np.array([1, 1, 1, 1, 1, 1, 1]), np.array([0, 1, 0, 1, 0, 1, 0]),
     np.array([1, 1, 1, 0, 0, 0, 0]), np.array([0, 0, 0, 0, 0, 0, 0]),
     np.array([1, 1, 1, 0, 1, 0, 1]))])
def test_find_would_be_ice(falling, cold, melting,  insect, result, iwc_cat_file):
    obj = _IceClassification(iwc_cat_file)
    obj.category_bits['falling'] = falling
    obj.category_bits['cold'] = cold
    obj.category_bits['melting'] = melting
    obj.category_bits['insect'] = insect
    testing.assert_array_equal(obj._find_would_be_ice().data, result)


@pytest.mark.parametrize("is_ice, attenuated, corrected, result", [
    (np.array([0, 0, 0, 1, 0, 1, 0]), np.array([1, 1, 0, 1, 0, 0, 1]),
     np.array([0, 0, 1, 1, 0, 1, 1]), np.array([0, 0, 0, 1, 0, 0, 0]))])
def test_find_corrected_ice(is_ice, attenuated, corrected, result, iwc_cat_file):
    obj = _IceClassification(iwc_cat_file)
    obj.quality_bits['attenuated'] = attenuated
    obj.quality_bits['corrected'] = corrected
    obj.is_ice = is_ice
    testing.assert_array_equal(obj._find_corrected_ice().data, result)


@pytest.mark.parametrize("is_ice, attenuated, corrected, result", [
    (np.array([0, 0, 0, 1, 0, 1, 0]), np.array([1, 1, 0, 1, 0, 1, 1]),
     np.array([0, 0, 1, 0, 0, 0, 1]), np.array([0, 0, 0, 1, 0, 1, 0]))])
def test_find_uncorrected_ice(is_ice, attenuated, corrected, result, iwc_cat_file):
    obj = _IceClassification(iwc_cat_file)
    obj.quality_bits['attenuated'] = attenuated
    obj.quality_bits['corrected'] = corrected
    obj.is_ice = is_ice
    testing.assert_array_equal(obj._find_uncorrected_ice().data, result)


@pytest.mark.parametrize("is_ice, is_rain, result", [
    (np.array([1, 0, 1]), np.array([0, 0, 1]),
     np.array([[0, 0, 0], [0, 0, 0], [1, 0, 1]]))])
def test_find_ice_above_rain(is_ice, is_rain, result, iwc_cat_file):
    obj = _IceClassification(iwc_cat_file)
    obj.is_ice = is_ice
    obj.is_rain = is_rain
    testing.assert_array_equal(obj._find_ice_above_rain().data, result)


@pytest.mark.parametrize("cold, is_rain, melting, result", [
    (np.array([1, 0, 1]), np.array([0, 0, 1]), np.array([1, 1, 0]),
     np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]))])
def test_find_cold_above_rain(cold, is_rain, melting, result, iwc_cat_file):
    obj = _IceClassification(iwc_cat_file)
    obj.category_bits['cold'] = cold
    obj.category_bits['melting'] = melting
    obj.is_rain = is_rain
    testing.assert_array_equal(obj._find_cold_above_rain().data, result)


def test_append_iwc_including_rain(iwc_source_file, iwc_cat_file):
    from cloudnetpy.products.iwc import _append_iwc_including_rain
    ice_class = _IceClassification(iwc_cat_file)
    ice_class.is_ice = np.array([0, 0, 1])
    ice_data = IwcSource(iwc_source_file)
    _append_iwc_including_rain(ice_data, ice_class)
    assert 'iwc_inc_rain' in ice_data.data.keys()


def test_append_iwc(iwc_source_file, iwc_cat_file):
    from cloudnetpy.products.iwc import _append_iwc
    ice_class = _IceClassification(iwc_cat_file)
    ice_class.ice_above_rain = np.array([0, 0, 1])
    ice_data = IwcSource(iwc_source_file)
    ice_data.data['iwc_inc_rain'] = [1, 1, 1]
    _append_iwc(ice_data, ice_class)
    assert 'iwc' in ice_data.data.keys()


def test_append_sensitivity(iwc_source_file):
    from cloudnetpy.products.iwc import _append_iwc_sensitivity
    ice_data = IwcSource(iwc_source_file)
    ice_data.mean_temperature = np.array([1.5])
    _append_iwc_sensitivity(ice_data)
    assert 'iwc_sensitivity' in ice_data.data.keys()


def test_append_bias(iwc_source_file):
    from cloudnetpy.products.iwc import _append_iwc_bias
    ice_data = IwcSource(iwc_source_file)
    _append_iwc_bias(ice_data)
    assert 'iwc_bias' in ice_data.data.keys()


# Create fake data to get all options for classification
class IceClass:
    def __init__(self):
        self.corrected_ice = np.asarray([[0, 0, 0, 1, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=bool)
        self.uncorrected_ice = np.asarray([[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0]], dtype=bool)
        self.is_ice = np.asarray([[0, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 0]], dtype=bool)
        self.cold_above_rain = np.asarray([[0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], dtype=bool)
        self.ice_above_rain = np.asarray([[0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0]], dtype=bool)
        self.would_be_ice = np.asarray([[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]], dtype=bool)


@pytest.mark.parametrize('value', [
    0, 1, 2, 3, 4, 5, 6, 7
])
def test_append_iwc_status(iwc_source_file, value):
    from cloudnetpy.products.iwc import _append_iwc_status
    ice_class = IceClass()
    ice_data = IwcSource(iwc_source_file)
    ice_data.data['iwc'] = np.ma.array([[1, 1, 1, 1, 1, 1],
                                        [1, 1, 1, 1, 1, 1]],
                                       mask=[[True, True, False, False, False, False],
                                             [True, True, True, False, False, False]])
    _append_iwc_status(ice_data, ice_class)
    assert value in ice_data.data['iwc_retrieval_status'][:]
