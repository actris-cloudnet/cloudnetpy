"""Module for reading ARM MMCR cloud radar data."""

import datetime
import logging
from os import PathLike
from uuid import UUID

import netCDF4
import numpy as np
from numpy import ma

from cloudnetpy import output, utils
from cloudnetpy.exceptions import ValidTimeStampError
from cloudnetpy.instruments.arm_utils import read_geolocation
from cloudnetpy.instruments.dealias import (
    CORRECTION_BITS_ATTRIBUTES,
    DEALIASED_V_ATTRIBUTES,
    add_correction_bits,
    dealias_velocity,
)
from cloudnetpy.instruments.instruments import MMCR
from cloudnetpy.instruments.nc_radar import NcRadar
from cloudnetpy.metadata import COMMON_ATTRIBUTES, MetaData

CDR_SNR_LIMIT = -5.0  # dB, co-polar SNR required for a valid CDR pixel
# Conservative fallback: the highest floor observed at SGP (2005). The
# floor drifted from -11 dB in early 2005 to -16...-19 dB from late 2005 on.
# Over-subtracting only weakens the depolarization signal, whereas
# under-subtracting turns rain and cloud into false insects.
CDR_FLOOR_DEFAULT = -11.0  # dB, cross-talk floor if it cannot be estimated
CDR_FLOOR_MARGIN = 1.5  # dB, added to the floor before subtraction
CDR_FLOOR_RANGE = 4000  # m, pixels above this are used to estimate the floor
CDR_FLOOR_MIN_PIXELS = 200
CDR_MAX_TIME_DIFF = 15  # s, max distance to the nearest dual-pol profile
LDR_MIN = -35.0  # dB


def mmcr2nc(
    raw_file: str | PathLike,
    output_file: str | PathLike,
    site_meta: dict,
    uuid: str | UUID | None = None,
    date: str | datetime.date | None = None,
) -> UUID:
    """Converts ARM MMCR cloud radar moments (mmcrmom) into Cloudnet Level 1b
    netCDF file.

    The MMCR cycles through several operating modes with different range
    resolutions and sensitivities. Only profiles from a single mode are used.
    Doppler velocities are folded at the Nyquist velocity of the mode, which
    is only about 5 m/s in the default general (GE) mode, and are dealiased
    using velocity continuity. Circular depolarization ratio is measured
    only in the dual-polarization mode (about one profile per 25 s). It is
    mapped to the selected mode by nearest time, corrected for the antenna
    cross-talk floor and stored as `ldr`.

    Args:
        raw_file: Daily ARM `mmcrmom` netCDF file, e.g.
            `sgpmmcrmomC1.b1.20100310.000047.cdf`.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key
            value pair is `name`. Optional are `latitude`, `longitude` and
            `altitude` (taken from the raw file if missing), `mode` (operating mode
            identifier, default = 'GE') and `snr_limit` (fixed SNR threshold in
            dB; by default the threshold is estimated from the noise in the top
            range gates).
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValidTimeStampError: No valid timestamps found.

    Examples:
          >>> from cloudnetpy.instruments import mmcr2nc
          >>> site_meta = {'name': 'Southern Great Plains'}
          >>> mmcr2nc('sgpmmcrmomC1.b1.20100310.000047.cdf', 'radar.nc', site_meta)

    """
    if isinstance(date, str):
        date = datetime.date.fromisoformat(date)
    uuid = utils.get_uuid(uuid)
    mode = site_meta.get("mode", "GE")
    snr_limit = site_meta.get("snr_limit")

    keymap = {
        "Reflectivity": "Zh",
        "MeanDopplerVelocity": "v",
        "SpectralWidth": "width",
        "SignalToNoiseRatio": "SNR",
    }

    with Mmcr(raw_file, site_meta) as mmcr:
        mmcr.init_data(keymap)
        mmcr.init_mode(mode)
        mmcr.screen_mode()
        mmcr.add_depolarization()
        if date is not None:
            mmcr.check_date(date)
        mmcr.sort_timestamps()
        mmcr.remove_duplicate_timestamps()
        mmcr.screen_by_snr(snr_limit)
        mmcr.mask_invalid_data()
        mmcr.flip_velocity_sign()
        mmcr.add_radar_specific_variables()
        mmcr.dealias_velocity()
        mmcr.add_zenith_angle()
        mmcr.add_site_geolocation()
        mmcr.add_height()
        mmcr.test_if_all_masked()
    attributes = output.add_time_attribute(ATTRIBUTES, mmcr.date)
    if mmcr.cdr_floor is not None:
        attributes["ldr"] = _ldr_attributes(mmcr.cdr_floor)
    output.update_attributes(mmcr.data, attributes)
    output.save_level1b(mmcr, output_file, uuid)
    return uuid


class Mmcr(NcRadar):
    """Class for ARM MMCR radar data. Child of NcRadar().

    Args:
        full_path: Filename of a daily ARM mmcrmom netCDF file.
        site_meta: Site properties in a dictionary. Required keys are: `name`.

    """

    def __init__(self, full_path: str | PathLike, site_meta: dict) -> None:
        super().__init__(full_path, {**site_meta})
        self.instrument = MMCR
        self.date = utils.get_epoch(self.dataset["time"].units).date()
        self.mode_index: int = 0
        self.cdr_floor: float | None = None
        self._add_geolocation_from_file()

    def init_mode(self, mode: str) -> None:
        """Adds time and range of the selected operating mode."""
        self.mode_index = self._find_mode_index(mode)
        n_heights = int(self.dataset["NumHeights"][self.mode_index])
        heights = self.dataset["heights"][self.mode_index, :n_heights]
        altitude = float(self.dataset["alt"][:])
        range_instru = np.array(heights - altitude)
        for cloudnet_array in self.data.values():
            if cloudnet_array.data.ndim == 2:
                cloudnet_array.data = cloudnet_array.data[:, :n_heights]
        self.append_data(range_instru, "range")
        self.append_data(np.array(self.time), "time")

    def screen_mode(self) -> None:
        """Keeps only profiles measured with the selected operating mode."""
        mode_num = np.array(self.dataset["ModeNum"][:])
        is_mode = mode_num == self.mode_index
        if not np.any(is_mode):
            msg = "No profiles found for the selected radar mode"
            raise ValidTimeStampError(msg)
        self.screen_time_indices(is_mode)

    def check_date(self, date: datetime.date) -> None:
        if self.date != date:
            raise ValidTimeStampError

    def add_radar_specific_variables(self) -> None:
        if self.instrument is None or self.instrument.frequency is None:
            msg = "Instrument not defined"
            raise RuntimeError(msg)
        self.append_data(self.instrument.frequency, "radar_frequency")
        nyquist = float(self.dataset["NyquistVelocity"][self.mode_index])
        self.append_data(nyquist, "nyquist_velocity")

    def flip_velocity_sign(self) -> None:
        # ARM: positive towards the radar. Cloudnet: positive away from the radar.
        self.data["v"].data *= -1

    def dealias_velocity(self) -> None:
        """Unfolds aliased Doppler velocities using continuity."""
        nyquist = float(self.data["nyquist_velocity"].data)
        self.data["v"].data = dealias_velocity(self.data["v"][:], nyquist)
        add_correction_bits(self.data)

    def add_zenith_angle(self) -> None:
        # MMCR is a fixed vertically pointing radar
        self.append_data(0.0, "zenith_angle")

    def add_depolarization(self) -> None:
        """Adds cross-talk corrected circular depolarization ratio as ldr.

        CDR is available only in the dual-polarization mode. Each profile of
        the selected mode gets the CDR of the nearest dual-pol profile in
        time. The antenna cross-talk floor, estimated as the median CDR of
        high-altitude pixels, is subtracted in linear units.
        """
        if "CircularDepolarizationRatio" not in self.dataset.variables:
            return
        try:
            ind_co, ind_cross = self._find_dualpol_modes()
        except ValueError:
            logging.info("No dual-polarization mode found, ldr not available")
            return
        mode_num = np.array(self.dataset["ModeNum"][:])
        time_raw = np.array(self.dataset["time"][:])
        co_ind = np.where(mode_num == ind_co)[0]
        cross_ind = np.where(mode_num == ind_cross)[0]
        if len(co_ind) == 0 or len(cross_ind) == 0:
            logging.info("No dual-polarization profiles found, ldr not available")
            return
        # Pair each cross-pol profile with the co-pol profile of the same pulse
        pair_ind = co_ind[_nearest_index(time_raw[co_ind], time_raw[cross_ind])]
        n_heights = int(self.dataset["NumHeights"][ind_cross])
        cdr = ma.masked_invalid(
            self.dataset["CircularDepolarizationRatio"][cross_ind, :n_heights]
        )
        snr = ma.masked_invalid(
            self.dataset["SignalToNoiseRatio"][pair_ind, :n_heights]
        )
        cdr[ma.filled(snr < CDR_SNR_LIMIT, fill_value=True)] = ma.masked
        altitude = float(self.dataset["alt"][:])
        range_cross = (
            np.array(self.dataset["heights"][ind_cross, :n_heights]) - altitude
        )
        self.cdr_floor = self._estimate_cdr_floor(cdr, range_cross)
        ldr = _remove_cdr_floor(cdr, self.cdr_floor)
        # Map to the profiles and range gates of the selected mode
        time_sel = np.array(self.data["time"][:]) * 3600
        time_ind = _nearest_index(time_raw[cross_ind], time_sel)
        range_ind = _nearest_index(range_cross, self.data["range"][:])
        ldr = ldr[time_ind][:, range_ind]
        too_far = np.abs(time_raw[cross_ind][time_ind] - time_sel) > CDR_MAX_TIME_DIFF
        ldr[too_far, :] = ma.masked
        self.append_data(ldr, "ldr")

    def _estimate_cdr_floor(
        self, cdr: ma.MaskedArray, range_cross: np.ndarray
    ) -> float:
        is_high = ~ma.getmaskarray(cdr) & (range_cross[np.newaxis, :] > CDR_FLOOR_RANGE)
        if np.count_nonzero(is_high) < CDR_FLOOR_MIN_PIXELS:
            logging.warning(
                "Not enough high-altitude pixels to estimate CDR floor, "
                "using conservative default %s dB",
                CDR_FLOOR_DEFAULT,
            )
            return CDR_FLOOR_DEFAULT
        floor = float(np.median(cdr[is_high].compressed()))
        logging.info("Estimated CDR cross-talk floor: %.1f dB", floor)
        return floor

    def _find_dualpol_modes(self) -> tuple[int, int]:
        """Returns indices of the co- and cross-polar dual-pol receiver modes.

        Mode names vary between files (e.g. "PO_Receiver0" in 2005,
        "DualPol_Receiver0" in 2008), so the receiver metadata is used.
        """
        for key in ("NumReceivers", "ReceiverNumber"):
            if key not in self.dataset.variables:
                msg = f"Variable {key} not found"
                raise ValueError(msg)
        n_receivers = ma.filled(self.dataset["NumReceivers"][:], 0)
        receiver = ma.filled(self.dataset["ReceiverNumber"][:], 0)
        co = np.where((n_receivers == 2) & (receiver == 1))[0]
        cross = np.where((n_receivers == 2) & (receiver == 2))[0]
        if len(co) != 1 or len(cross) != 1:
            msg = "Dual-polarization modes not found"
            raise ValueError(msg)
        return int(co[0]), int(cross[0])

    def _find_mode_index(self, mode: str) -> int:
        var = self.dataset["ModeDescription"]
        var.set_auto_mask(False)
        names = [str(d).strip() for d in netCDF4.chartostring(var[:])]
        for ind, name in enumerate(names):
            if name.endswith(f"_{mode}"):
                logging.info("Using radar mode %s", name)
                return ind
        available = [name for name in names if name and "Reserved" not in name]
        msg = f"Radar mode '{mode}' not found. Available modes: {available}"
        raise ValueError(msg)

    def _add_geolocation_from_file(self) -> None:
        self.site_meta = read_geolocation(self.dataset, self.site_meta)


def _nearest_index(reference: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Returns index of the nearest `reference` value for each `query` value."""
    if len(reference) < 2:
        return np.zeros(len(query), dtype=int)
    order = np.argsort(reference)
    sorted_ref = reference[order]
    pos = np.searchsorted(sorted_ref, query)
    pos = np.clip(pos, 1, len(sorted_ref) - 1)
    use_left = np.abs(query - sorted_ref[pos - 1]) <= np.abs(sorted_ref[pos] - query)
    return order[np.where(use_left, pos - 1, pos)]


def _remove_cdr_floor(cdr: ma.MaskedArray, floor: float) -> ma.MaskedArray:
    """Subtracts the cross-talk floor in linear units, clipping at LDR_MIN."""
    linear = 10 ** (cdr / 10) - 10 ** ((floor + CDR_FLOOR_MARGIN) / 10)
    ldr = ma.masked_where(linear <= 0, linear)
    ldr = 10 * ma.log10(ldr)
    ldr = ldr.filled(LDR_MIN)
    ldr = np.maximum(ldr, LDR_MIN)
    return ma.masked_where(ma.getmaskarray(cdr), ldr)


def _ldr_attributes(floor: float) -> MetaData:
    return COMMON_ATTRIBUTES["ldr"]._replace(
        comment=(
            "Circular depolarization ratio from the dual-polarization mode, "
            "mapped to the selected mode by nearest time. The antenna cross-talk "
            f"floor of {floor:.1f} dB has been subtracted in linear units, "
            f"and values are clipped at {LDR_MIN} dB."
        ),
    )


ATTRIBUTES = {
    "correction_bits": CORRECTION_BITS_ATTRIBUTES,
    "v": DEALIASED_V_ATTRIBUTES,
    "zenith_angle": COMMON_ATTRIBUTES["zenith_angle"]._replace(dimensions=None),
    "nyquist_velocity": COMMON_ATTRIBUTES["nyquist_velocity"]._replace(dimensions=None),
}
