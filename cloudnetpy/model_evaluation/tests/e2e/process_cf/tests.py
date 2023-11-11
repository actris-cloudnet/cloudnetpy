import netCDF4
import pytest


class TestCloudFractionProcessing:
    product = "cf"

    @pytest.fixture(autouse=True)
    def _fetch_params(self, params) -> None:
        self.full_path = params["full_path"]

    @pytest.mark.reprocess()
    def test_that_has_correct_attributes(self) -> None:
        nc = netCDF4.Dataset(self.full_path)
        assert nc.location == "Mace-Head"
        assert nc.year == "2019"
        assert nc.month == "05"
        assert nc.day == "17"
        assert nc.title == "Downsampled Cf of ecmwf from Mace-Head"
        assert nc.cloudnet_file_type == "l3-cf"
        assert nc.Conventions == "CF-1.8"
        assert (
            nc.source
            == "Observation file: 20190517_mace-head_categorize.nc\necmwf file(s): 20190517_mace-head_ecmwf.nc"
        )
        nc.close()

    @pytest.mark.reprocess()
    @pytest.mark.parametrize(
        "key",
        ["cf_V_ecmwf", "cf_A_ecmwf", "cf_V_adv_ecmwf", "cf_A_adv_ecmwf"],
    )
    def test_that_has_correct_product_variables(self, key) -> None:
        nc = netCDF4.Dataset(self.full_path)
        assert key in nc.variables
        nc.close()

    @pytest.mark.reprocess()
    @pytest.mark.parametrize(
        "key",
        ["time", "level", "latitude", "longitude", "horizontal_resolution"],
    )
    def test_that_has_correct_model_variables(self, key) -> None:
        nc = netCDF4.Dataset(self.full_path)
        assert key in nc.variables
        nc.close()

    @pytest.mark.reprocess()
    @pytest.mark.parametrize(
        "key",
        ["ecmwf_forecast_time", "ecmwf_height", "ecmwf_cf", "ecmwf_cf_cirrus"],
    )
    def test_that_has_correct_cycle_variables(self, key) -> None:
        nc = netCDF4.Dataset(self.full_path)
        assert key in nc.variables
        nc.close()
