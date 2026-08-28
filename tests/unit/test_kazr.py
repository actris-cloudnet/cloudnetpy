import shutil
from os import path
from tempfile import TemporaryDirectory

import netCDF4
import numpy as np
import pytest
from numpy import ma

from cloudnetpy.exceptions import RadarDataError, ValidTimeStampError
from cloudnetpy.instruments import kazr
from cloudnetpy.instruments.dealias import dealias_velocity
from tests.unit.all_products_fun import Check
from tests.unit.radar_fun import RadarFun

SCRIPT_PATH = path.dirname(path.realpath(__file__))
FILEPATH = f"{SCRIPT_PATH}/data/kazr/"
FILES = [
    f"{FILEPATH}/sgpkazrcfrgeC1.a1.20220601.000007.nc",
    f"{FILEPATH}/sgpkazrcfrgeC1.a1.20220601.010009.nc",
]


class TestKazr2nc(Check):
    site_meta = {
        "name": "Southern Great Plains",
        "latitude": 36.605,
        "longitude": -97.485,
        "altitude": 318,
    }
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/kazr.nc"
    uuid = kazr.kazr2nc(FILEPATH, temp_path, site_meta)
    date = "2022-06-01"

    def test_variable_names(self):
        keys = {
            "Zh",
            "v",
            "width",
            "ldr",
            "SNR",
            "correction_bits",
            "time",
            "range",
            "radar_frequency",
            "nyquist_velocity",
            "snr_limit",
            "latitude",
            "longitude",
            "altitude",
            "zenith_angle",
            "height",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_variables(self):
        assert np.isclose(
            self.nc.variables["radar_frequency"][:].data, 34.89, atol=0.01
        )
        assert np.isclose(
            self.nc.variables["nyquist_velocity"][:].data, 5.96, atol=0.01
        )
        assert self.nc.variables["zenith_angle"][:] == 0
        assert -20 < self.nc.variables["snr_limit"][:] < -10

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_global_attributes(self):
        assert self.nc.source == "ProSensing KAZR"
        assert self.nc.title == f"KAZR cloud radar from {self.site_meta['name']}"

    def test_range(self):
        assert self.nc.dimensions["range"].size == 678
        assert np.isclose(self.nc.variables["range"][0], 100.68, atol=0.01)
        assert np.allclose(
            self.nc.variables["height"][:],
            self.nc.variables["range"][:] + self.site_meta["altitude"],
        )

    def test_time_concatenation(self):
        time = self.nc.variables["time"][:]
        assert len(time) == 80
        assert np.all(np.diff(time) > 0)
        assert time[0] < 0.01
        assert 1.0 < time[-1] < 1.1

    def test_correction_bits(self):
        bits = self.nc.variables["correction_bits"][:]
        v_mask = ma.getmaskarray(self.nc.variables["v"][:])
        assert np.all(bits[~v_mask] == 1)
        assert np.array_equal(ma.getmaskarray(bits), v_mask)

    def test_velocity_sign(self):
        # Falling ice cloud should have negative velocity
        v = self.nc.variables["v"][:]
        height = self.nc.variables["range"][:]
        ice = v[:, (height > 6000) & (height < 10000)]
        assert ma.median(ice) < -0.3

    def test_ldr(self):
        # Ice is at the leakage floor which is subtracted -> minimum value
        ldr = self.nc.variables["ldr"][:]
        height = self.nc.variables["range"][:]
        ice = ldr[:, (height > 6000) & (height < 10000)]
        assert ma.median(ice) == -30
        assert "leakage floor of -2" in self.nc.variables["ldr"].comment

    def test_snr_screening(self):
        snr = self.nc.variables["SNR"][:]
        assert ma.min(snr) >= self.nc.variables["snr_limit"][:] - 3
        for key in ("Zh", "v", "width", "ldr"):
            mask = ma.getmaskarray(self.nc.variables[key][:])
            assert np.all(mask[ma.getmaskarray(snr)])

    def test_file_list_and_single_file(self, tmp_path):
        test_path = tmp_path / "list.nc"
        kazr.kazr2nc(FILES, test_path, self.site_meta, date=self.date)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.dimensions["time"].size == 80
        kazr.kazr2nc(FILES[0], test_path, self.site_meta, date=self.date)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.dimensions["time"].size == 40

    def test_next_day_file_screened(self, tmp_path):
        next_day = tmp_path / "sgpkazrcfrgeC1.a1.20220602.000009.nc"
        shutil.copy(FILES[1], next_day)
        with netCDF4.Dataset(next_day, "a") as nc:
            nc["time"].units = "seconds since 2022-06-02 00:00:09 0:00"
        test_path = tmp_path / "next.nc"
        kazr.kazr2nc([FILES[0], next_day], test_path, self.site_meta, date=self.date)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.dimensions["time"].size == 40
            assert np.all(nc.variables["time"][:] < 1)

    def test_invalid_file_skipped(self, tmp_path, caplog):
        invalid = tmp_path / "sgpkazrcfrgeC1.a1.20220601.010009.nc"
        shutil.copy(FILES[1], invalid)
        with netCDF4.Dataset(invalid, "a") as nc:
            nc.renameVariable("linear_depolarization_ratio", "foo")
            nc.createVariable("linear_depolarization_ratio", "f4", ("time",))
        test_path = tmp_path / "skipped.nc"
        kazr.kazr2nc([FILES[0], invalid], test_path, self.site_meta, date=self.date)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.dimensions["time"].size == 40
        assert "Skipping file" in caplog.text

    def test_missing_nyquist_velocity(self, tmp_path):
        invalid = tmp_path / "sgpkazrcfrgeC1.a1.20220601.000007.nc"
        shutil.copy(FILES[0], invalid)
        with netCDF4.Dataset(invalid, "a") as nc:
            nc.renameVariable("nyquist_velocity", "foo")
        with pytest.raises(RadarDataError, match="Nyquist"):
            kazr.kazr2nc(invalid, tmp_path / "x.nc", self.site_meta, date=self.date)

    def test_geolocation_from_file(self, tmp_path):
        test_path = tmp_path / "geo.nc"
        kazr.kazr2nc(FILEPATH, test_path, {"name": "SGP"})
        with netCDF4.Dataset(test_path) as nc:
            assert np.isclose(nc.variables["latitude"][0], 36.605)
            assert np.isclose(nc.variables["longitude"][0], -97.485)
            assert np.isclose(nc.variables["altitude"][0], 318)

    def test_wrong_date_validation(self, tmp_path):
        test_path = tmp_path / "invalid.nc"
        with pytest.raises(ValidTimeStampError):
            kazr.kazr2nc(FILEPATH, test_path, self.site_meta, date="2022-06-02")

    def test_uuid_from_user(self, tmp_path):
        test_path = tmp_path / "uuid.nc"
        uuid_from_user = "fe45561b-eb08-4d2a-a463-c6b4f7be7055"
        uuid = kazr.kazr2nc(FILEPATH, test_path, self.site_meta, uuid=uuid_from_user)
        with netCDF4.Dataset(test_path) as nc:
            assert nc.file_uuid == uuid_from_user
            assert str(uuid) == uuid_from_user


class TestKazrCorrected2nc(Check):
    site_meta = {
        "name": "Southern Great Plains",
        "latitude": 36.605,
        "longitude": -97.485,
        "altitude": 318,
    }
    temp_dir = TemporaryDirectory()
    temp_path = temp_dir.name + "/kazr-cor.nc"
    filepath = f"{SCRIPT_PATH}/data/kazr-cor/sgpkazrcorgeC1.c1.20120601.000001.nc"
    uuid = kazr.kazr2nc(filepath, temp_path, site_meta)
    date = "2012-06-01"

    def test_variable_names(self):
        keys = {
            "Zh",
            "v",
            "width",
            "ldr",
            "SNR",
            "correction_bits",
            "time",
            "range",
            "radar_frequency",
            "nyquist_velocity",
            "latitude",
            "longitude",
            "altitude",
            "zenith_angle",
            "height",
        }
        assert set(self.nc.variables.keys()) == keys

    def test_variables(self):
        # Parsed from global attributes
        assert np.isclose(self.nc.variables["radar_frequency"][:].data, 34.83)
        assert np.isclose(
            self.nc.variables["nyquist_velocity"][:].data, 5.96, atol=0.01
        )

    def test_common_radar(self):
        radar_fun = RadarFun(self.nc, self.site_meta, self.date, self.uuid)
        for name, method in RadarFun.__dict__.items():
            if "test_" in name:
                getattr(radar_fun, name)()

    def test_time(self):
        time = self.nc.variables["time"][:]
        assert len(time) == 60
        assert np.all(np.diff(time) > 0)
        assert 16 < time[0] < 16.01

    def test_ldr_from_cross_pol(self):
        ldr = self.nc.variables["ldr"][:]
        height = self.nc.variables["range"][:]
        rain = ldr[:, (height > 1000) & (height < 2500)]
        assert -30 <= ma.median(rain) < -20
        melting = ldr[:, (height > 3100) & (height < 3500)]
        assert ma.median(melting) > ma.median(rain) + 4

    def test_velocity_sign(self):
        v = self.nc.variables["v"][:]
        height = self.nc.variables["range"][:]
        rain = v[:, (height > 1000) & (height < 2500)]
        assert ma.median(rain) < -3

    def test_detection_mask_screening(self):
        with netCDF4.Dataset(self.filepath) as nc:
            mask = nc.variables["significant_detection_mask"][:]
        z_mask = ma.getmaskarray(self.nc.variables["Zh"][:])
        assert np.all(z_mask[mask != 1])  # ARM noise is masked
        assert np.mean(~z_mask[mask == 1]) > 0.95  # small speckle removed


class TestDealiasVelocity:
    nyquist = 6.0

    def test_folded_rain_profile(self):
        # True velocities: ice -1 m/s at top, increasing fall speed to -8 m/s
        true = np.linspace(-8, -1, 50)[np.newaxis, :].repeat(3, axis=0)
        folded = np.where(true < -self.nyquist, true + 2 * self.nyquist, true)
        assert np.any(folded > 0)
        result = dealias_velocity(ma.array(folded), self.nyquist)
        assert np.allclose(result, true)

    def test_unaliased_data_unchanged(self):
        v = ma.array(np.random.default_rng(0).uniform(-3, 3, (5, 30)))
        result = dealias_velocity(v, self.nyquist)
        assert np.allclose(result, v)

    def test_gap_uses_previous_profile(self):
        true = np.full((2, 40), -8.0)
        true[:, 20:] = -1.0  # ice above, rain below with a gap in between
        folded = np.where(true < -self.nyquist, true + 2 * self.nyquist, true)
        v = ma.array(folded)
        v[:, 5:20] = ma.masked  # gap larger than max_gap
        v[0, :5] = -5.0  # first profile: unaliased rain to seed time continuity
        result = dealias_velocity(v, self.nyquist, max_gap=5)
        assert np.allclose(result[1, :5], -8.0)
        assert ma.getmaskarray(result)[1, 10]

    def test_masked_values_preserved(self):
        v = ma.array(np.full((2, 10), -1.0))
        v[0, 3] = ma.masked
        result = dealias_velocity(v, self.nyquist)
        assert ma.getmaskarray(result)[0, 3]
        assert not ma.getmaskarray(result)[1, 3]

    def test_isolated_wrong_profile_fixed_by_neighbours(self):
        true = np.full((30, 20), -8.0)
        folded = true + 2 * self.nyquist  # +4 m/s everywhere (all folded)
        v = ma.array(folded)
        v[:, 10:] = ma.masked  # no unaliased anchor above the rain
        v[:, :10][0] = -8.0  # only the first profile is known good
        result = dealias_velocity(v, self.nyquist)
        assert np.allclose(result[:, :10], -8.0)


class TestLdrScreening:
    site_meta = {"name": "SGP", "altitude": 318}

    def test_broken_cross_polar_channel_removes_ldr(self, tmp_path, caplog):
        filepath = f"{SCRIPT_PATH}/data/kazr-andoya/"
        test_path = tmp_path / "andoya.nc"
        kazr.kazr2nc(filepath, test_path, {"name": "Andoya"})
        assert "Cross-polar channel unreliable" in caplog.text
        with netCDF4.Dataset(test_path) as nc:
            assert "ldr" not in nc.variables
            assert "SNRx" not in nc.variables

    def test_healthy_channel_keeps_ldr(self, tmp_path):
        test_path = tmp_path / "sgp.nc"
        kazr.kazr2nc(f"{SCRIPT_PATH}/data/kazr/", test_path, self.site_meta)
        with netCDF4.Dataset(test_path) as nc:
            assert "ldr" in nc.variables
            assert "SNRx" not in nc.variables

    def test_low_cross_polar_snr_masked_in_corrected_data(self, tmp_path):
        raw = f"{SCRIPT_PATH}/data/kazr-cor/sgpkazrcorgeC1.c1.20120601.000001.nc"
        test_path = tmp_path / "cor.nc"
        kazr.kazr2nc(raw, test_path, self.site_meta)
        with netCDF4.Dataset(raw) as nc:
            snrx = nc.variables["signal_to_noise_ratio_xpol"][:]
            detected = nc.variables["significant_detection_mask"][:] == 1
        with netCDF4.Dataset(test_path) as nc:
            ldr_mask = ma.getmaskarray(nc.variables["ldr"][:])
            z_mask = ma.getmaskarray(nc.variables["Zh"][:])
        assert np.all(ldr_mask[detected & (snrx < -10)])
        assert not np.all(ldr_mask[detected & (snrx >= -10)])
        assert np.all(ldr_mask[z_mask])


class TestLdrFloor:
    def test_floor_found(self):
        rng = np.random.default_rng(0)
        ldr = ma.array(
            np.concatenate([rng.normal(-20, 0.3, 5000), rng.uniform(-30, -5, 2000)])
        )
        floor = kazr._find_ldr_floor(ldr)
        assert floor is not None
        assert abs(floor + 20) < 0.3

    def test_no_floor_in_broad_distribution(self):
        rng = np.random.default_rng(0)
        assert kazr._find_ldr_floor(ma.array(rng.uniform(-30, -5, 5000))) is None

    def test_floor_outside_plausible_range_ignored(self):
        rng = np.random.default_rng(0)
        assert kazr._find_ldr_floor(ma.array(rng.normal(-35, 0.3, 5000))) is None

    def test_floor_found_when_insects_dominate(self):
        rng = np.random.default_rng(0)
        ldr = ma.array(
            np.concatenate([rng.normal(-20, 0.3, 1500), rng.uniform(-15, -3, 8500)])
        )
        floor = kazr._find_ldr_floor(ldr)
        assert floor is not None
        assert abs(floor + 20) < 0.3
