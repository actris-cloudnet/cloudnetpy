"""Radar module, containing the :class:`Radar` class."""

import logging
import math

import numpy as np
from numpy import ma
from scipy import constants

from cloudnetpy import utils
from cloudnetpy.categorize.attenuations import RadarAttenuation
from cloudnetpy.constants import GHZ_TO_HZ, SEC_IN_HOUR, SPEED_OF_LIGHT
from cloudnetpy.datasource import DataSource


class Radar(DataSource):
    """Radar class, child of DataSource.

    Args:
        full_path: Cloudnet Level 1 radar netCDF file.

    Attributes:
        radar_frequency (float): Radar frequency (GHz).
        folding_velocity (float): Radar's folding velocity (m/s).
        location (str): Location of the radar, copied from the global attribute
            `location` of the input file.
        sequence_indices (list): Indices denoting the different altitude
            regimes of the radar.
        source_type (str): Type of the radar, copied from the global attribute
            `source` of the *radar_file*. Can be free form string but must
            include either 'rpg' or 'mira' denoting one of the two supported
            radars.

    See Also:
        :func:`instruments.rpg2nc()`, :func:`instruments.mira2nc()`

    """

    def __init__(self, full_path: str):
        super().__init__(full_path, radar=True)
        self.radar_frequency = float(self.getvar("radar_frequency"))
        self.folding_velocity = self._get_folding_velocity()
        self.sequence_indices = self._get_sequence_indices()
        self.location = getattr(self.dataset, "location", "")
        self.source_type = getattr(self.dataset, "source", "")
        self.height: np.ndarray
        self.altitude: float
        self._init_data()
        self._init_sigma_v()
        self._get_folding_velocity_full()

    def rebin_to_grid(self, time_new: np.ndarray) -> list:
        """Rebins radar data in time using mean.

        Args:
            time_new: Target time array as fraction hour. Updates *time* attribute.

        """
        bad_time_indices = []
        for key, array in self.data.items():
            match key:
                case "ldr" | "sldr" | "Z":
                    array.db2lin()
                    bad_time_indices = array.rebin_data(self.time, time_new)
                    array.lin2db()
                case "v":
                    array.rebin_velocity(
                        self.time,
                        time_new,
                        self.folding_velocity,
                        self.sequence_indices,
                    )
                case "v_sigma":
                    array.calc_linear_std(self.time, time_new)
                case "width":
                    array.rebin_data(self.time, time_new)
                case "rainfall_rate":
                    array.rebin_data(self.time, time_new, mask_zeros=False)
                case _:
                    continue
        return bad_time_indices

    def remove_incomplete_pixels(self) -> None:
        """Mask radar pixels where one or more required quantities are missing.

        All valid radar pixels **must** contain proper values for `Z`, and `v` and
        also for `width` if exists. Otherwise there is some kind of problem with the
        data and the pixel should not be used in any further analysis.

        """
        good_ind = ~ma.getmaskarray(self.data["Z"][:]) & ~ma.getmaskarray(
            self.data["v"][:],
        )

        if "width" in self.data:
            good_ind = good_ind & ~ma.getmaskarray(self.data["width"][:])

        for array in self.data.values():
            if array.data.ndim == 2:
                array.mask_indices(~good_ind)

    def filter_speckle_noise(self) -> None:
        """Removes speckle noise from radar data.

        Any isolated radar pixel, i.e. "hot pixel", is assumed to
        exist due to speckle noise. This is a crude approach and a
        more sophisticated method could be implemented here later.

        """
        for key in ("Z", "v", "width", "ldr", "v_sigma"):
            if key in self.data:
                self.data[key].filter_vertical_stripes()

    def filter_1st_gate_artifact(self) -> None:
        """Removes 1st range gate velocity artifact."""
        velocity_limit = 4
        ind = np.where(self.data["v"][:, 0] > velocity_limit)
        self.data["v"][:][ind, 0] = ma.masked

    def filter_stripes(self, variable: str) -> None:
        """Filters vertical and horizontal stripe-shaped artifacts from radar data."""
        if variable not in self.data:
            return
        data = ma.copy(self.data[variable][:])
        n_points_in_profiles = ma.count(data, axis=1)
        n_profiles_with_data = np.count_nonzero(n_points_in_profiles)
        if n_profiles_with_data < 300:
            return
        n_vertical = self._filter(
            data,
            axis=1,
            min_coverage=0.5,
            z_limit=10,
            distance=4,
            n_blocks=100,
        )
        n_horizontal = self._filter(
            data,
            axis=0,
            min_coverage=0.3,
            z_limit=-30,
            distance=3,
            n_blocks=20,
        )
        if n_vertical > 0 or n_horizontal > 0:
            logging.debug(
                "Filtered %s vertical and %s horizontal stripes "
                "from radar data using %s",
                n_vertical,
                n_horizontal,
                variable,
            )

    def _filter(
        self,
        data: np.ndarray,
        axis: int,
        min_coverage: float,
        z_limit: float,
        distance: float,
        n_blocks: int,
    ) -> int:
        if axis == 0:
            data = data.T
            echo = self.data["Z"][:].T
        else:
            echo = self.data["Z"][:]

        len_block = int(np.floor(data.shape[0] / n_blocks))
        block_indices = np.arange(len_block)
        n_removed_total = 0

        for block_number in range(n_blocks):
            data_block = data[block_indices, :]
            n_values = ma.count(data_block, axis=1)
            try:
                q1 = np.quantile(n_values, 0.25)
                q3 = np.quantile(n_values, 0.75)
            except IndexError:
                continue

            if q1 == q3:
                continue

            threshold = distance * (q3 - q1) + q3

            indices = np.where(
                (n_values > threshold) & (n_values > (min_coverage * data.shape[1])),
            )[0]
            true_ind = [int(x) for x in (block_number * len_block + indices)]
            n_removed = len(indices)

            if n_removed > 5:
                continue

            if n_removed > 0:
                n_removed_total += n_removed
                for ind in true_ind:
                    ind2 = np.where(echo[ind, :] < z_limit)
                    bad_indices = (ind, ind2) if axis == 1 else (ind2, ind)
                    self.data["v"][:][bad_indices] = ma.masked
            block_indices += len_block

        return n_removed_total

    def correct_atten(self, attenuations: RadarAttenuation) -> None:
        """Corrects radar echo for liquid and gas attenuation.

        Args:
            attenuations: Radar attenuation object.

        References:
            The method is based on Hogan R. and O'Connor E., 2004,
            https://bit.ly/2Yjz9DZ and the original Cloudnet Matlab implementation.

        """
        z_corrected = self.data["Z"][:] + attenuations.gas.amount
        ind = ma.where(attenuations.liquid.amount)
        z_corrected[ind] += attenuations.liquid.amount[ind]
        ind = ma.where(attenuations.rain.amount)
        z_corrected[ind] += attenuations.rain.amount[ind]
        ind = ma.where(attenuations.melting.amount)
        z_corrected[ind] += attenuations.melting.amount[ind]
        self.append_data(z_corrected, "Z")

    def calc_errors(
        self,
        attenuations: RadarAttenuation,
        is_clutter: np.ndarray,
    ) -> None:
        """Calculates uncertainties of radar echo.

        Calculates and adds `Z_error`, `Z_sensitivity` and `Z_bias`
        :class:`CloudnetArray` instances to `data` attribute.

        Args:
            attenuations: 2-D attenuations due to atmospheric gases.
            is_clutter: 2-D boolean array denoting pixels contaminated by clutter.

        References:
            The method is based on Hogan R. and O'Connor E., 2004,
            https://bit.ly/2Yjz9DZ and the original Cloudnet Matlab implementation.

        """

        def _calc_sensitivity() -> np.ndarray:
            """Returns sensitivity of radar as function of altitude."""
            mean_gas_atten = ma.mean(attenuations.gas.amount, axis=0)
            z_sensitivity = z_power_min + log_range + mean_gas_atten
            zc = ma.median(ma.array(z, mask=~is_clutter), axis=0)
            valid_values = np.logical_not(zc.mask)
            z_sensitivity[valid_values] = zc[valid_values]
            return z_sensitivity

        def _calc_error() -> np.ndarray | float:
            """Returns error of radar as function of altitude.

            References:
                Hogan, R. J., 1998: Dual-wavelength radar studies of clouds.
                PhD Thesis, University of Reading, UK.

            """
            noise_threshold = 3
            n_pulses = _number_of_independent_pulses()
            ln_to_log10 = 10 / np.log(10)
            z_precision = ma.divide(ln_to_log10, np.sqrt(n_pulses)) * (
                1 + (utils.db2lin(z_power_min - z_power) / noise_threshold)
            )

            z_error = utils.l2norm(
                z_precision,
                attenuations.liquid.error.filled(0),
                attenuations.rain.error.filled(0),
                attenuations.melting.error.filled(0),
            )

            z_error[
                attenuations.liquid.uncorrected
                | attenuations.rain.uncorrected
                | attenuations.melting.uncorrected
            ] = ma.masked

            return z_error

        def _number_of_independent_pulses() -> float:
            """Returns number of independent pulses.

            References:
                Atlas, D., 1964: Advances in radar meteorology.
                Advances in Geophys., 10, 318-478.

            """
            if "width" not in self.data:
                default_width = 0.3
                width = np.zeros_like(self.data["Z"][:])
                width[~width.mask] = default_width
            else:
                width = self.data["width"][:]
            dwell_time = utils.mdiff(self.time) * SEC_IN_HOUR
            wl = SPEED_OF_LIGHT / (self.radar_frequency * GHZ_TO_HZ)
            return 4 * np.sqrt(math.pi) * dwell_time * width / wl

        def _calc_z_power_min() -> float:
            if ma.all(z_power.mask):
                return 0
            return np.percentile(z_power.compressed(), 0.1)

        z = self.data["Z"][:]
        radar_range = self.to_m(self.dataset.variables["range"])
        log_range = utils.lin2db(radar_range, scale=20)
        z_power = z - log_range
        z_power_min = _calc_z_power_min()
        self.append_data(_calc_error(), "Z_error")
        self.append_data(_calc_sensitivity(), "Z_sensitivity")
        self.append_data(1.0, "Z_bias")

    def add_meta(self) -> None:
        """Copies misc. metadata from the input file."""
        for key in ("latitude", "longitude", "altitude"):
            self.append_data(np.array(self.getvar(key)), key)
        for key in ("time", "height", "radar_frequency"):
            self.append_data(np.array(getattr(self, key)), key)

    def _init_data(self) -> None:
        self.append_data(self.getvar("Zh"), "Z", units="dBZ")
        for key in ("v", "ldr", "width", "sldr", "rainfall_rate"):
            try:
                self._variables_to_cloudnet_arrays((key,))
            except KeyError:
                continue

    def _init_sigma_v(self) -> None:
        """Initializes std of the velocity field. The std will be calculated
        later when re-binning the data.
        """
        self.append_data(self.getvar("v"), "v_sigma")

    def _get_sequence_indices(self) -> list:
        """Mira has only one sequence and one folding velocity. RPG has
        several sequences with different folding velocities.
        """
        if self.height is None:
            msg = "Height not found in the input file"
            raise RuntimeError(msg)
        all_indices = np.arange(len(self.height))
        if not utils.isscalar(self.folding_velocity):
            starting_indices = self.getvar("chirp_start_indices")
            return np.split(all_indices, starting_indices[1:])
        return [all_indices]

    def _get_folding_velocity(self) -> np.ndarray | float:
        if "nyquist_velocity" in self.dataset.variables:
            return self.getvar("nyquist_velocity")
        if "prf" in self.dataset.variables:
            prf = self.getvar("prf")
            return _prf_to_folding_velocity(prf, self.radar_frequency)
        msg = "Unable to determine folding velocity"
        raise RuntimeError(msg)

    def _get_folding_velocity_full(self) -> None:
        folding_velocity: list | np.ndarray = []
        if utils.isscalar(self.folding_velocity):
            folding_velocity = np.repeat(
                self.folding_velocity,
                len(self.sequence_indices[0]),
            )
        else:
            folding_velocity = list(folding_velocity)
            self.folding_velocity = np.array(self.folding_velocity)
            for indices, velocity in zip(
                self.sequence_indices,
                self.folding_velocity,
                strict=True,
            ):
                folding_velocity.append(np.repeat(velocity, len(indices)))
            folding_velocity = np.hstack(folding_velocity)
        self.append_data(folding_velocity, "nyquist_velocity")


def _prf_to_folding_velocity(prf: np.ndarray, radar_frequency: float) -> float:
    ghz_to_hz = 1e9
    if len(prf) != 1:
        msg = "Unable to determine folding velocity"
        raise RuntimeError(msg)
    return float(prf[0] * constants.c / (4 * radar_frequency * ghz_to_hz))
