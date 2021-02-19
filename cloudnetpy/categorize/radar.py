"""Radar module, containing the :class:`Radar` class."""
import math
from typing import Union
import numpy as np
import numpy.ma as ma
from scipy import constants
from cloudnetpy.categorize import ProfileDataSource
from cloudnetpy.categorize.classify import ClassificationResult
from cloudnetpy import utils


class Radar(ProfileDataSource):
    """Radar class, child of ProfileDataSource.

    Args:
        full_path: Cloudnet Level 1 radar netCDF file.

    Attributes:
        radar_frequency (float): Radar frequency (GHz).
        folding_velocity (float): Radar's folding velocity (m/s).
        location (str): Location of the radar, copied from the global attribute `location` of the
            input file.
        sequence_indices (list): Indices denoting the different altitude
            regimes of the radar.
        type (str): Type of the radar, copied from the global attribute
            `source` of the *radar_file*. Can be free form string but must
            include either 'rpg' or 'mira' denoting one of the two supported
            radars.

    See Also:
        :func:`instruments.rpg2nc()`, :func:`instruments.mira2nc()`

    """
    def __init__(self, full_path: str):
        super().__init__(full_path, radar=True)
        self.radar_frequency = float(self.getvar('radar_frequency', 'frequency'))
        self.folding_velocity = self._get_folding_velocity()
        self.sequence_indices = self._get_sequence_indices()
        self.location = getattr(self.dataset, 'location', '')
        self.type = getattr(self.dataset, 'source', '')
        self._init_data()
        self._init_sigma_v()

    def rebin_to_grid(self, time_new: np.ndarray) -> None:
        """Rebins radar data in time using mean.

        Args:
            time_new: Target time array as fraction hour. Updates *time* attribute.

        """
        for key in self.data:
            if key in ('Z', 'ldr'):
                self.data[key].db2lin()
                self.data[key].rebin_data(self.time, time_new)
                self.data[key].lin2db()
            elif key == 'v':
                # This has some problems with RPG data when folding is present.
                self.data[key].rebin_velocity(self.time, time_new,
                                              self.folding_velocity,
                                              self.sequence_indices)
            elif key == 'v_sigma':
                self.data[key].calc_linear_std(self.time, time_new)
            else:
                self.data[key].rebin_data(self.time, time_new)
        self.time = time_new

    def remove_incomplete_pixels(self) -> None:
        """Mask radar pixels where one or more required quantities are missing.

        All valid radar pixels **must** contain proper values for `Z`, and `v` and
        also for `width` if exists. Otherwise there is some kind of problem with the
        data and the pixel should not be used in any further analysis.

        """
        good_ind = (~ma.getmaskarray(self.data['Z'][:])
                    & ~ma.getmaskarray(self.data['v'][:]))

        if 'width' in self.data.keys():
            good_ind = good_ind & ~ma.getmaskarray(self.data['width'][:])

        for array in self.data.values():
            array.mask_indices(~good_ind)

    def filter_speckle_noise(self) -> None:
        """Removes speckle noise from radar data.

        Any isolated radar pixel, i.e. "hot pixel", is assumed to
        exist due to speckle noise. This is a crude approach and a
        more sophisticated method could be implemented here later.

        See Also:
            :func:`utils.filter_isolated_pixels()`

        """
        for key in ('Z', 'v', 'width', 'ldr', 'v_sigma'):
            if key in self.data.keys():
                self.data[key].filter_isolated_pixels()

    def correct_atten(self, attenuations: dict) -> None:
        """Corrects radar echo for liquid and gas attenuation.

        Args:
            attenuations: 2-D attenuations due to atmospheric gases and liquid: `radar_gas_atten`,
                `radar_liquid_atten`.

        References:
            The method is based on Hogan R. and O'Connor E., 2004, https://bit.ly/2Yjz9DZ
            and the original Cloudnet Matlab implementation.

        """
        z_corrected = self.data['Z'][:] + attenuations['radar_gas_atten']
        ind = ma.where(attenuations['radar_liquid_atten'])
        z_corrected[ind] += attenuations['radar_liquid_atten'][ind]
        self.append_data(z_corrected, 'Z')

    def calc_errors(self, attenuations: dict, classification: ClassificationResult) -> None:
        """Calculates uncertainties of radar echo.

        Calculates and adds `Z_error`, `Z_sensitivity` and `Z_bias` :class:`CloudnetArray`
        instances to `data` attribute.

        Args:
            attenuations: 2-D attenuations due to atmospheric gases.
            classification: The :class:`ClassificationResult` instance.

        References:
            The method is based on Hogan R. and O'Connor E., 2004, https://bit.ly/2Yjz9DZ
            and the original Cloudnet Matlab implementation.

        """
        def _calc_sensitivity() -> np.ndarray:
            """Returns sensitivity of radar as function of altitude."""
            mean_gas_atten = ma.mean(attenuations['radar_gas_atten'], axis=0)
            z_sensitivity = z_power_min + log_range + mean_gas_atten
            zc = ma.median(ma.array(z, mask=~classification.is_clutter), axis=0)
            z_sensitivity[~zc.mask] = zc[~zc.mask]
            return z_sensitivity

        def _calc_error() -> np.ndarray:
            if 'width' not in self.data:
                return 0.3
            z_precision = 4.343 * (1 / np.sqrt(_number_of_independent_pulses())
                                   + utils.db2lin(z_power_min - z_power) / 3)
            gas_error = attenuations['radar_gas_atten'] * 0.1
            liq_error = attenuations['liquid_atten_err'].filled(0)
            z_error = utils.l2norm(gas_error, liq_error, z_precision)
            z_error[attenuations['liquid_uncorrected']] = ma.masked
            return z_error

        def _number_of_independent_pulses() -> float:
            seconds_in_hour = 3600
            dwell_time = utils.mdiff(self.time) * seconds_in_hour
            return (dwell_time * self.radar_frequency * 1e9 * 4
                    * np.sqrt(math.pi) * self.data['width'][:] / 3e8)

        def _calc_z_power_min() -> float:
            if ma.all(z_power.mask):
                return 0
            return np.percentile(z_power.compressed(), 0.1)

        z = self.data['Z'][:]
        radar_range = self.km2m(self.dataset.variables['range'])
        log_range = utils.lin2db(radar_range, scale=20)
        z_power = z - log_range
        z_power_min = _calc_z_power_min()
        self.append_data(_calc_error(), 'Z_error')
        self.append_data(_calc_sensitivity(), 'Z_sensitivity')
        self.append_data(1, 'Z_bias')

    def add_meta(self) -> None:
        """Copies misc. metadata from the input file."""
        for key in ('latitude', 'longitude', 'altitude'):
            self.append_data(self.getvar(key), key)
        for key in ('time', 'height', 'radar_frequency'):
            self.append_data(getattr(self, key), key)

    def _init_data(self):
        self._unknown_variable_to_cloudnet_array(('Ze',), 'Z', units='dBZ')
        for key in ('v', 'ldr', 'width'):
            try:
                self._variables_to_cloudnet_arrays((key,))
            except KeyError:
                continue

    def _init_sigma_v(self) -> None:
        """Initializes std of the velocity field. The std will be calculated
        later when re-binning the data."""
        self.append_data(self.getvar('v'), 'v_sigma')

    def _get_sequence_indices(self) -> list:
        """Mira has only one sequence and one folding velocity. RPG has
        several sequences with different folding velocities."""
        all_indices = np.arange(len(self.height))
        if not utils.isscalar(self.folding_velocity):
            starting_indices = self.getvar('chirp_start_indices')
            return np.split(all_indices, starting_indices[1:])
        return [all_indices]

    def _get_folding_velocity(self) -> Union[np.ndarray, float]:
        for key in ('nyquist_velocity', 'NyquistVelocity'):
            if key in self.dataset.variables:
                return self.getvar(key)
        if 'prf' in self.dataset.variables:
            prf = self.getvar('prf')
            return _prf_to_folding_velocity(prf, self.radar_frequency)
        raise RuntimeError('Unable to determine folding velocity')


def _prf_to_folding_velocity(prf: np.ndarray, radar_frequency: float) -> float:
    ghz_to_hz = 1e9
    return float(prf * constants.c / (4*radar_frequency*ghz_to_hz))
