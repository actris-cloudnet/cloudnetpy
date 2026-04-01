import os.path

import netCDF4

from cloudnetpy.instruments.weather_radar import wr2nc

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

SITE_META = {
    "name": "Test Site",
    "latitude": 60.0,
    "longitude": 25.0,
    "altitude": 100.0,
}


class TestWr2nc:
    def test_wr2nc(self, tmp_path):
        input_file = f"{SCRIPT_PATH}/data/wr/202603160005_radar.polar.fianj_ZDRCAL.h5"
        output_file = tmp_path / "test_output.nc"
        file_uuid = wr2nc(
            input_files=input_file,
            output_file=output_file,
            site_meta=SITE_META,
        )
        assert output_file.exists()
        assert file_uuid is not None
        with netCDF4.Dataset(output_file) as nc:
            assert "time" in nc.variables
            assert "range" in nc.variables
            assert "height" in nc.variables
            assert "SNR" in nc.variables
            assert "v" in nc.variables
            assert "width" in nc.variables
            assert "zdr" in nc.variables
            assert "rho_hv" in nc.variables
            assert "radar_frequency" in nc.variables

            assert "time" in nc.dimensions
            assert "range" in nc.dimensions

            time_var = nc.variables["time"]
            assert len(time_var) > 0

            range_var = nc.variables["range"]
            assert len(range_var) > 0
