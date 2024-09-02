from dataclasses import dataclass
from typing import Annotated

import numpy as np
from numpy import ma
from numpy.typing import NDArray

from cloudnetpy import constants as con
from cloudnetpy.utils import path_lengths_from_ground


@dataclass
class Attenuation:
    amount: Annotated[ma.MaskedArray, "float32"]
    error: Annotated[ma.MaskedArray, "float32"]
    attenuated: NDArray[np.bool_]
    uncorrected: NDArray[np.bool_]


@dataclass
class RadarAttenuation:
    gas: Attenuation
    liquid: Attenuation
    rain: Attenuation
    melting: Attenuation


def calc_two_way_attenuation(
    height: np.ndarray, specific_attenuation: ma.MaskedArray
) -> ma.MaskedArray:
    """Calculates two-way attenuation (dB) for given specific attenuation
    (dB km-1) and height (m).
    """
    path_lengths = path_lengths_from_ground(height) * con.M_TO_KM  # km
    one_way_attenuation = specific_attenuation * path_lengths
    accumulated_attenuation = ma.cumsum(one_way_attenuation, axis=1)
    return accumulated_attenuation * con.TWO_WAY
