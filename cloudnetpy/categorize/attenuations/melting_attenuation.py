import numpy as np
from numpy import ma

import cloudnetpy.constants as con
from cloudnetpy import utils
from cloudnetpy.categorize.attenuations import (
    Attenuation,
)
from cloudnetpy.categorize.containers import ClassificationResult, Observations


def calc_melting_attenuation(
    data: Observations, classification: ClassificationResult
) -> Attenuation:
    shape = classification.category_bits.melting.shape
    is_rain = classification.is_rain

    affected_region = classification.category_bits.freezing.copy()

    if data.disdrometer is None:
        affected_region[~is_rain, :] = False
        return Attenuation(
            amount=ma.masked_all(shape),
            error=ma.masked_all(shape),
            attenuated=affected_region,
            uncorrected=affected_region,
        )

    rainfall_rate = data.disdrometer.data["rainfall_rate"][:]
    frequency = data.radar.radar_frequency

    attenuation_array = _calc_melting_attenuation(rainfall_rate, frequency)

    amount = affected_region * utils.transpose(attenuation_array)

    affected_region[amount == 0] = False

    amount[amount == 0] = ma.masked

    band = utils.get_wl_band(data.radar.radar_frequency)
    error_factor = 0.2 if band == 0 else 0.1

    error = amount * error_factor
    error[~affected_region] = ma.masked

    return Attenuation(
        amount=amount,
        error=error,
        attenuated=affected_region,
        uncorrected=affected_region & amount.mask,
    )


def _calc_melting_attenuation(
    rainfall_rate: np.ndarray, frequency: float
) -> np.ndarray:
    """Calculates total attenuation due to melting layer (dB).

    References:
        Li, H., & Moisseev, D. (2019). Melting layer attenuation
        at Ka- and W-bands as derived from multifrequency radar
        Doppler spectra observations. Journal of Geophysical
        Research: Atmospheres, 124, 9520–9533. https://doi.org/10.1029/2019JD030316

    """
    if frequency > 34 and frequency < 37:
        a, b = 0.97, 0.61
    elif frequency > 93 and frequency < 96:
        a, b = 2.9, 0.42
    else:
        msg = "Radar frequency not supported"
        raise ValueError(msg)
    return a * (rainfall_rate * con.M_S_TO_MM_H) ** b