"""Module for dealiasing Doppler velocities of vertically pointing radars."""

import warnings

import numpy as np
from numpy import ma

from cloudnetpy.cloudnetarray import CloudnetArray
from cloudnetpy.metadata import COMMON_ATTRIBUTES, MetaData
from cloudnetpy.utils import bit_field_definition

DEALIASED_V_ATTRIBUTES = COMMON_ATTRIBUTES["v"]._replace(
    comment=(COMMON_ATTRIBUTES["v"].comment or "")
    + "\nDealiased using velocity continuity along the profile and in time."
)

CORRECTION_BITS_ATTRIBUTES = MetaData(
    long_name="Correction bits",
    units="1",
    definition=bit_field_definition({0: """Doppler velocity is dealiased."""}),
    comment=(
        "This parameter is a bit field that indicates which corrections have\n"
        "been applied to radar measurements."
    ),
    dimensions=("time", "range"),
)


def add_correction_bits(data: dict) -> None:
    """Adds correction bits indicating dealiased Doppler velocity."""
    v = data["v"][:]
    bits = ma.ones(v.shape, dtype=np.uint32)
    bits.mask = ma.getmaskarray(v)
    data["correction_bits"] = CloudnetArray(bits, "correction_bits")


def dealias_velocity(
    velocity: ma.MaskedArray,
    nyquist: float,
    n_ref: int = 5,
    max_gap: int = 10,
    n_neighbours: int = 10,
) -> ma.MaskedArray:
    """Unfolds Doppler velocity of a vertically pointing radar.

    Each profile is processed from the top gate downwards. For each gate, the
    aliasing interval (-1, 0 or +1 times 2 * Nyquist velocity) closest to the
    median of the last `n_ref` valid gates above is selected. If there is no
    valid gate within `max_gap` gates above, the previous profile near the
    same gate is used as the reference, and if that is also missing, the
    interval closest to -nyquist / 2 is selected (targets are assumed to fall).
    Finally, each gate is checked against the median of the surrounding
    `n_neighbours` profiles to remove isolated wrongly unfolded profiles.

    Args:
        velocity: Velocity array (time, range) with positive values upwards.
        nyquist: Nyquist velocity (m/s).
        n_ref: Number of gates used for the reference.
        max_gap: Maximum gap (in gates) over which the reference is valid.
        n_neighbours: Number of profiles on both sides used in the final check.

    Returns:
        Dealiased velocity.

    """
    v: ma.MaskedArray = ma.array(velocity, dtype=float, copy=True)
    mask = ma.getmaskarray(v)
    data = np.array(ma.filled(v, np.nan))
    n_time, n_range = data.shape
    buffer = np.full((n_time, n_ref), np.nan)  # last valid unfolded values above
    ref_gate = np.full(n_time, -1)
    shifts = 2 * nyquist * np.array([-1.0, 0.0, 1.0])
    for gate in range(n_range - 1, -1, -1):
        valid = ~mask[:, gate]
        has_ref = valid & (ref_gate - gate <= max_gap) & ~np.isnan(buffer[:, 0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # All-NaN slices
            reference = np.nanmedian(buffer, axis=1)
        # Unfold profiles with a reference from the gates above
        candidates = data[has_ref, gate][:, np.newaxis] + shifts
        best = np.argmin(np.abs(candidates - reference[has_ref, np.newaxis]), axis=1)
        data[has_ref, gate] = candidates[np.arange(len(best)), best]
        # Others: reference from previous unfolded profile, else assume falling
        no_ref = np.where(valid & ~has_ref)[0]
        buffer[no_ref] = np.nan
        gates = slice(gate, gate + max_gap + 1)
        for ind in no_ref:
            previous = data[ind - 1, gates] if ind > 0 else np.array([np.nan])
            previous = previous[~np.isnan(previous)]
            ref = np.median(previous) if previous.size else -nyquist / 2
            options = data[ind, gate] + shifts
            data[ind, gate] = options[np.argmin(np.abs(options - ref))]
        # Update rolling reference buffer
        buffer[valid, 1:] = buffer[valid, :-1]
        buffer[valid, 0] = data[valid, gate]
        ref_gate[valid] = gate
    _fix_outlier_profiles(data, nyquist, n_neighbours)
    v[~mask] = data[~mask]
    return v


def _fix_outlier_profiles(data: np.ndarray, nyquist: float, n_neighbours: int) -> None:
    """Shifts values that differ from the neighbouring profiles by more than
    the Nyquist velocity.
    """
    n_time, n_range = data.shape
    if n_time < 2 * n_neighbours + 1:
        return
    padded = np.pad(
        data, ((n_neighbours, n_neighbours), (0, 0)), constant_values=np.nan
    )
    shifts = 2 * nyquist * np.array([-1.0, 0.0, 1.0])
    for gate in range(n_range):
        column = data[:, gate]
        valid = ~np.isnan(column)
        if not np.any(valid):
            continue
        windows = np.lib.stride_tricks.sliding_window_view(
            padded[:, gate], 2 * n_neighbours + 1
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(windows, axis=1)
        is_outlier = valid & (np.abs(column - median) > nyquist)
        if not np.any(is_outlier):
            continue
        candidates = column[is_outlier][:, np.newaxis] + shifts
        best = np.argmin(np.abs(candidates - median[is_outlier, np.newaxis]), axis=1)
        data[is_outlier, gate] = candidates[np.arange(len(best)), best]
