import netCDF4
import numpy as np
import pytest
from numpy import ma
from numpy.testing import assert_array_almost_equal, assert_array_equal

from cloudnetpy import constants
from cloudnetpy.products import cod
from cloudnetpy.products.cod import OpticalDepthSource, generate_cod

# category bits: droplet=1, falling=2, freezing=4
DROPLET, FALLING, FREEZING = 1, 2, 4
# quality bits: radar=1, lidar=2
RADAR, LIDAR = 1, 2
# adiabatic LWP consistent with the fixture LWP of 0.1 kg m-2
LWP_ADIABATIC = np.array([0, 0.1, 0, 0.1])
LWC_REL_ERROR = np.full((4, 4), 0.2)
DER_REL_ERROR = np.full((4, 4), 0.3)
IWC_ERROR_DB = np.full((4, 4), 1.7)


@pytest.fixture(scope="session")
def categorize_file(tmpdir_factory):
    file_name = tmpdir_factory.mktemp("data").join("categorize.nc")
    _create_categorize_file(file_name, with_lwp=True)
    return str(file_name)


@pytest.fixture(scope="session")
def categorize_file_no_mwr(tmpdir_factory):
    file_name = tmpdir_factory.mktemp("data").join("categorize_no_mwr.nc")
    _create_categorize_file(file_name, with_lwp=False)
    return str(file_name)


def _create_categorize_file(file_name, *, with_lwp: bool) -> None:
    n_time, n_height = 4, 4
    with netCDF4.Dataset(file_name, "w", format="NETCDF4_CLASSIC") as nc:
        for name, n in (
            ("time", n_time),
            ("height", n_height),
            ("model_time", n_time),
            ("model_height", n_height),
        ):
            nc.createDimension(name, n)
            var = nc.createVariable(name, "f8", name)
            var[:] = np.arange(n)
        nc.variables["height"][:] = [100, 200, 300, 400]
        nc.variables["height"].units = "m"
        nc.variables["model_height"][:] = [100, 200, 300, 400]
        var = nc.createVariable("altitude", "f8")
        var[:] = 0
        var.units = "m"
        nc.createVariable("radar_frequency", "f8")[:] = 35.5
        if with_lwp:
            nc.createVariable("lwp", "f8", "time")[:] = [0.1, 0.1, 0.1, 0.1]
            nc.createVariable("lwp_error", "f8", "time")[:] = [0.01, 0.01, 0.01, 0.01]
        # profile 0: clear, 1: liquid + ice, 2: ice only, 3: liquid in rain
        nc.createVariable("rainfall_rate", "f8", "time")[:] = [0, 0, 0, 1]
        cat = np.zeros((n_time, n_height), dtype=int)
        cat[1, 0:2] = DROPLET
        cat[1, 2:4] = FALLING | FREEZING
        cat[2, 1:4] = FALLING | FREEZING
        cat[3, 0:2] = DROPLET
        nc.createVariable("category_bits", "i4", ("time", "height"))[:] = cat
        qual = np.full((n_time, n_height), RADAR | LIDAR, dtype=int)
        nc.createVariable("quality_bits", "i4", ("time", "height"))[:] = qual
        nc.createVariable("temperature", "f8", ("model_time", "model_height"))[:] = (
            np.full((n_time, n_height), 260.0)
        )
        nc.createVariable("pressure", "f8", ("model_time", "model_height"))[:] = (
            np.full((n_time, n_height), 90000.0)
        )
        nc.createVariable("Z", "f8", ("time", "height"))[:] = np.full(
            (n_time, n_height), -10.0
        )
        nc.createVariable("Z_error", "f8", ("time", "height"))[:] = np.full(
            (n_time, n_height), 1.0
        )
        nc.createVariable("is_rain", "i4", "time")[:] = [0, 0, 0, 1]
        nc.year, nc.month, nc.day = "2025", "06", "10"
        nc.location = "Kumpula"
        nc.file_uuid = "b7d3e2f0-1234-5678-9abc-def012345678"
        nc.cloudnet_file_type = "categorize"


class TestOpticalDepthSource:
    @pytest.fixture(autouse=True)
    def run_before_tests(self, categorize_file):
        self.obj = OpticalDepthSource(categorize_file, assumed_der=10e-6)
        yield
        self.obj.close()

    def test_liquid_extinction_formula(self, monkeypatch):
        lwc = ma.array(np.full((4, 4), 1e-3), mask=~self.obj.is_liquid)
        der = ma.array(np.full((4, 4), 20e-6), mask=np.zeros((4, 4), dtype=bool))
        monkeypatch.setattr(
            self.obj, "_get_lwc", lambda: (lwc, LWC_REL_ERROR, LWP_ADIABATIC)
        )
        monkeypatch.setattr(
            self.obj, "_get_der", lambda: (der, ma.array(DER_REL_ERROR))
        )
        self.obj.append_liquid_extinction()
        ext = self.obj.data["extinction_liquid"][:]
        expected = 3 * 1e-3 / (2 * 1000 * 20e-6)
        assert_array_almost_equal(ext[1, 0:2], expected)
        assert ext[0, :].mask.all()
        assert ext[1, 2:4].mask.all()
        assert not self.obj.is_der_assumed.any()

    def test_liquid_extinction_assumed_der(self, monkeypatch):
        lwc = ma.array(np.full((4, 4), 1e-3), mask=~self.obj.is_liquid)
        der = ma.masked_all((4, 4))
        der[1, 0] = 20e-6
        der[1, 1] = 100e-6  # outside the valid range
        monkeypatch.setattr(
            self.obj, "_get_lwc", lambda: (lwc, LWC_REL_ERROR, LWP_ADIABATIC)
        )
        monkeypatch.setattr(
            self.obj, "_get_der", lambda: (der, ma.array(DER_REL_ERROR))
        )
        self.obj.append_liquid_extinction()
        ext = self.obj.data["extinction_liquid"][:]
        assert_array_almost_equal(ext[1, 0], 3 * 1e-3 / (2 * 1000 * 20e-6))
        assert_array_almost_equal(ext[1, 1], 3 * 1e-3 / (2 * 1000 * 10e-6))
        assert_array_equal(self.obj.is_der_assumed[1, :], [False, True, False, False])
        assert_array_equal(self.obj.is_der_assumed[3, :], [True, True, False, False])

    def test_extended_cloud_top_pixel_is_kept(self, monkeypatch):
        # lwc retrieval placed liquid in pixel (1, 2), which has no droplet bit
        lwc = ma.array(np.full((4, 4), 1e-3), mask=~self.obj.is_liquid)
        lwc[1, 2] = 1e-3
        der = ma.masked_all((4, 4))
        monkeypatch.setattr(
            self.obj, "_get_lwc", lambda: (lwc, LWC_REL_ERROR, LWP_ADIABATIC)
        )
        monkeypatch.setattr(
            self.obj, "_get_der", lambda: (der, ma.array(DER_REL_ERROR))
        )
        self.obj.append_liquid_extinction()
        ext = self.obj.data["extinction_liquid"][:]
        assert ext[1, 2] > 0
        assert ext[1, 3] is ma.masked
        assert (DER_REL_ERROR == 0.3).all()  # input constant not mutated

    def test_ice_extinction_formula(self, monkeypatch):
        is_ice = self.obj.ice_classification.is_ice
        iwc = ma.array(np.full((4, 4), 1e-4), mask=~is_ice)
        ier = ma.array(np.full((4, 4), 50e-6), mask=~is_ice)
        iwc[2, 1] = ma.masked  # lidar-only ice pixel
        monkeypatch.setattr(self.obj, "_get_iwc", lambda: (iwc, ma.array(IWC_ERROR_DB)))
        monkeypatch.setattr(self.obj, "_get_ier", lambda: ier)
        self.obj.append_ice_extinction()
        ext = self.obj.data["extinction_ice"][:]
        expected = 3 * 1e-4 / (2 * constants.RHO_ICE * 50e-6)
        assert_array_almost_equal(ext[1, 2:4], expected)
        assert_array_almost_equal(ext[2, 2:4], expected)
        assert ext[2, 1] is ma.masked
        assert ext[0, :].mask.all()
        assert_array_equal(
            self.obj.is_lidar_only_ice[2, :], [False, True, False, False]
        )

    def test_ice_masked_in_rain(self, monkeypatch):
        is_ice = self.obj.ice_classification.is_ice
        iwc = ma.array(np.full((4, 4), 1e-4), mask=~is_ice)
        ier = ma.array(np.full((4, 4), 50e-6), mask=~is_ice)
        self.obj.is_rain = np.array([False, False, True, False])
        monkeypatch.setattr(self.obj, "_get_iwc", lambda: (iwc, ma.array(IWC_ERROR_DB)))
        monkeypatch.setattr(self.obj, "_get_ier", lambda: ier)
        self.obj.append_ice_extinction()
        lwc = ma.masked_all((4, 4))
        monkeypatch.setattr(
            self.obj, "_get_lwc", lambda: (lwc, LWC_REL_ERROR, LWP_ADIABATIC)
        )
        monkeypatch.setattr(self.obj, "_get_der", lambda: (lwc, lwc))
        self.obj.append_liquid_extinction()
        self.obj.append_extinction_status()
        assert self.obj.data["extinction_ice"][:][2, :].mask.all()
        assert_array_equal(self.obj.data["extinction_retrieval_status"][:][2, 1:], 6)

    def test_optical_depths_and_status(self, monkeypatch):
        is_ice = self.obj.ice_classification.is_ice
        lwc = ma.array(np.full((4, 4), 1e-3), mask=~self.obj.is_liquid)
        lwc[3, :] = ma.masked  # rain profile has no lwc
        der = ma.array(np.full((4, 4), 20e-6), mask=~self.obj.is_liquid)
        der[1, 1] = ma.masked
        iwc = ma.array(np.full((4, 4), 1e-4), mask=~is_ice)
        iwc[2, 1] = ma.masked
        ier = ma.array(np.full((4, 4), 50e-6), mask=~is_ice)
        monkeypatch.setattr(
            self.obj, "_get_lwc", lambda: (lwc, LWC_REL_ERROR, LWP_ADIABATIC)
        )
        monkeypatch.setattr(
            self.obj, "_get_der", lambda: (der, ma.array(DER_REL_ERROR))
        )
        monkeypatch.setattr(self.obj, "_get_iwc", lambda: (iwc, ma.array(IWC_ERROR_DB)))
        monkeypatch.setattr(self.obj, "_get_ier", lambda: ier)
        self.obj.append_liquid_extinction()
        self.obj.append_ice_extinction()
        self.obj.append_extinction_status()
        self.obj.append_optical_depths()
        self.obj.append_optical_depth_error()
        self.obj.append_optical_depth_status()

        ext_liq = 3 * 1e-3 / (2 * 1000 * 20e-6)
        ext_liq_assumed = 3 * 1e-3 / (2 * 1000 * 10e-6)
        ext_ice = 3 * 1e-4 / (2 * constants.RHO_ICE * 50e-6)
        dz = 100

        tau_liq = self.obj.data["optical_depth_liquid"][:]
        tau_ice = self.obj.data["optical_depth_ice"][:]
        tau = self.obj.data["optical_depth"][:]
        assert_array_almost_equal(
            tau_liq[0:3], [0, (ext_liq + ext_liq_assumed) * dz, 0]
        )
        assert_array_almost_equal(tau_ice[0:3], [0, 2 * ext_ice * dz, 2 * ext_ice * dz])
        assert_array_almost_equal(tau[0:3], tau_liq[0:3] + tau_ice[0:3])
        assert tau_liq[3] is ma.masked
        assert tau_ice[3] is ma.masked
        assert tau[3] is ma.masked

        pixel_status = self.obj.data["extinction_retrieval_status"][:]
        assert_array_equal(pixel_status[0, :], [0, 0, 0, 0])
        assert_array_equal(pixel_status[1, :], [1, 2, 3, 3])
        assert_array_equal(pixel_status[2, :], [0, 6, 3, 3])
        assert_array_equal(pixel_status[3, :], [6, 6, 0, 0])

        status = self.obj.data["optical_depth_retrieval_status"][:]
        assert_array_equal(status, [0, 2, 3, 5])

        # errors: 10 log10(1 + relative error)
        err_liq = self.obj.data["extinction_liquid_error"][:]
        assert_array_almost_equal(err_liq[1, 0], 10 * np.log10(1 + np.hypot(0.2, 0.3)))
        assert_array_almost_equal(err_liq[1, 1], 10 * np.log10(1 + np.hypot(0.2, 0.4)))
        assert err_liq[0, :].mask.all()
        err_ice = self.obj.data["extinction_ice_error"][:]
        assert_array_almost_equal(err_ice[2, 2:4], 1.7)
        assert err_ice[2, 1] is ma.masked
        err = self.obj.data["optical_depth_error"][:]
        rel_ice = 10 ** (1.7 / 10) - 1
        rel_liq = np.array([np.hypot(0.2, 0.3), np.hypot(0.2, 0.4)])
        w = np.array([ext_liq, ext_liq_assumed])
        rel_liq_col = np.sum(rel_liq * w) / np.sum(w)
        abs_err = np.hypot(rel_liq_col * tau_liq[1], rel_ice * tau_ice[1])
        assert_array_almost_equal(err[1], 10 * np.log10(1 + abs_err / tau[1]))
        assert_array_almost_equal(err[2], 1.7)
        assert err[0] is ma.masked
        assert err[3] is ma.masked

    def test_corrected_ice_status(self, monkeypatch):
        is_ice = self.obj.ice_classification.is_ice
        lwc = ma.masked_all((4, 4))
        der = ma.masked_all((4, 4))
        iwc = ma.array(np.full((4, 4), 1e-4), mask=~is_ice)
        ier = ma.array(np.full((4, 4), 50e-6), mask=~is_ice)
        corrected = np.zeros((4, 4), dtype=bool)
        corrected[2, 2] = True
        monkeypatch.setattr(self.obj.ice_classification, "corrected_ice", corrected)
        monkeypatch.setattr(
            self.obj, "_get_lwc", lambda: (lwc, LWC_REL_ERROR, LWP_ADIABATIC)
        )
        monkeypatch.setattr(
            self.obj, "_get_der", lambda: (der, ma.array(DER_REL_ERROR))
        )
        monkeypatch.setattr(self.obj, "_get_iwc", lambda: (iwc, ma.array(IWC_ERROR_DB)))
        monkeypatch.setattr(self.obj, "_get_ier", lambda: ier)
        self.obj.append_liquid_extinction()
        self.obj.append_ice_extinction()
        self.obj.append_extinction_status()
        self.obj.append_optical_depths()
        self.obj.append_optical_depth_status()
        assert self.obj.data["extinction_retrieval_status"][:][2, 2] == 4
        assert self.obj.data["optical_depth_retrieval_status"][:][2] == 1
        # liquid present without lwc -> no retrieval
        assert self.obj.data["optical_depth_retrieval_status"][:][1] == 5
        assert self.obj.data["optical_depth"][:][1] is ma.masked

    def test_inconsistent_lwp_status(self, monkeypatch):
        is_ice = self.obj.ice_classification.is_ice
        lwc = ma.array(np.full((4, 4), 1e-3), mask=~self.obj.is_liquid)
        der = ma.array(np.full((4, 4), 20e-6), mask=~self.obj.is_liquid)
        iwc = ma.array(np.full((4, 4), 1e-4), mask=~is_ice)
        ier = ma.array(np.full((4, 4), 50e-6), mask=~is_ice)
        # profile 1 can hold only 0.02 kg m-2 but LWP is 0.1 kg m-2
        lwp_adiabatic = np.array([0, 0.02, 0, 0.1])
        monkeypatch.setattr(
            self.obj, "_get_lwc", lambda: (lwc, LWC_REL_ERROR, lwp_adiabatic)
        )
        monkeypatch.setattr(
            self.obj, "_get_der", lambda: (der, ma.array(DER_REL_ERROR))
        )
        monkeypatch.setattr(self.obj, "_get_iwc", lambda: (iwc, ma.array(IWC_ERROR_DB)))
        monkeypatch.setattr(self.obj, "_get_ier", lambda: ier)
        self.obj.append_liquid_extinction()
        self.obj.append_ice_extinction()
        self.obj.append_extinction_status()
        self.obj.append_optical_depths()
        self.obj.append_optical_depth_status()
        assert_array_equal(self.obj.is_lwp_inconsistent, [False, True, False, False])
        status = self.obj.data["optical_depth_retrieval_status"][:]
        assert_array_equal(status, [0, 4, 1, 5])
        # value is kept, only flagged
        assert self.obj.data["optical_depth"][:][1] > 0


def test_generate_cod(categorize_file, tmp_path):
    output_file = tmp_path / "optical_depth.nc"
    uuid = generate_cod(categorize_file, output_file)
    with netCDF4.Dataset(output_file) as nc:
        assert nc.file_uuid == str(uuid)
        assert nc.cloudnet_file_type == "cod"
        for key in (
            "extinction_liquid",
            "extinction_ice",
            "extinction_retrieval_status",
            "optical_depth_liquid",
            "optical_depth_ice",
            "optical_depth",
            "optical_depth_retrieval_status",
            "extinction_liquid_error",
            "extinction_ice_error",
            "optical_depth_error",
            "lwp",
        ):
            assert key in nc.variables
        assert nc.variables["extinction_liquid"].dimensions == ("time", "height")
        assert nc.variables["optical_depth"].dimensions == ("time",)
        assert nc.variables["optical_depth"].units == "1"
        assert "10 um" in nc.variables["extinction_liquid"].comment
        tau = nc.variables["optical_depth"][:]
        status = nc.variables["optical_depth_retrieval_status"][:]
        assert tau[0] == 0
        assert status[0] == 0
        assert tau[1] > 0
        assert tau[2] > 0
        assert tau[3] is ma.masked
        assert status[3] == 5
        # thin fixture layer cannot hold 0.1 kg m-2 adiabatically
        assert status[1] == 4
        ext_liq = nc.variables["extinction_liquid"][:]
        assert ext_liq[1, 0:2].count() == 2
        ext_ice = nc.variables["extinction_ice"][:]
        assert ext_ice[1, 2:4].count() == 2
        assert ext_ice[2, 1:4].count() == 3


def test_non_positive_lwp_gives_no_liquid_retrieval(categorize_file, tmp_path):
    src = OpticalDepthSource(categorize_file, assumed_der=10e-6)
    src.dataset.variables["lwp"][:]  # LWP is 0.1 in the fixture
    lwc, _, _ = src._get_lwc()
    assert not lwc[1, 0:2].mask.any()
    src.close()
    with netCDF4.Dataset(categorize_file, "a") as nc:
        nc.variables["lwp"][1] = 0.0
    try:
        output_file = tmp_path / "optical_depth.nc"
        generate_cod(categorize_file, output_file)
        with netCDF4.Dataset(output_file) as nc:
            assert nc.variables["optical_depth"][:][1] is ma.masked
            assert nc.variables["optical_depth_retrieval_status"][:][1] == 5
            assert_array_equal(
                nc.variables["extinction_retrieval_status"][:][1, 0:2], 6
            )
    finally:
        with netCDF4.Dataset(categorize_file, "a") as nc:
            nc.variables["lwp"][1] = 0.1


def test_generate_without_mwr(categorize_file_no_mwr, tmp_path):
    output_file = tmp_path / "optical_depth.nc"
    generate_cod(categorize_file_no_mwr, output_file)
    with netCDF4.Dataset(output_file) as nc:
        assert "lwp" not in nc.variables
        assert nc.variables["extinction_liquid"][:].mask.all()
        tau = nc.variables["optical_depth"][:]
        status = nc.variables["optical_depth_retrieval_status"][:]
        assert tau[2] > 0  # ice-only profile still retrieved
        assert status[2] == 1
        assert tau[1] is ma.masked  # liquid present, no LWP
        assert status[1] == 5


def test_generate_cod_custom_der(categorize_file, tmp_path):
    output_file = tmp_path / "optical_depth.nc"
    generate_cod(categorize_file, output_file, assumed_der=5e-6)
    with netCDF4.Dataset(output_file) as nc:
        assert "5 um" in nc.variables["extinction_liquid"].comment


def test_attributes_cover_all_variables(categorize_file, tmp_path):
    output_file = tmp_path / "optical_depth.nc"
    generate_cod(categorize_file, output_file)
    with netCDF4.Dataset(output_file) as nc:
        for key in cod.OPTICAL_DEPTH_ATTRIBUTES:
            assert nc.variables[key].long_name
