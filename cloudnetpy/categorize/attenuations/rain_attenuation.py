import numpy as np
from numpy import ma
from numpy.typing import NDArray

import cloudnetpy.constants as con
from cloudnetpy import utils
from cloudnetpy.categorize.attenuations import (
    Attenuation,
    calc_two_way_attenuation,
)
from cloudnetpy.categorize.containers import ClassificationResult, Observations


def calc_rain_attenuation(
    data: Observations, classification: ClassificationResult
) -> Attenuation:
    affected_region, inducing_region = _find_regions(classification)
    shape = affected_region.shape

    if data.disdrometer is None:
        return Attenuation(
            amount=ma.masked_all(shape),
            error=ma.masked_all(shape),
            attenuated=affected_region,
            uncorrected=affected_region,
        )

    rainfall_rate = data.disdrometer.data["rainfall_rate"][:].copy()
    rainfall_rate[classification.is_rain == 0] = ma.masked
    frequency = data.radar.radar_frequency

    specific_attenuation_array = _calc_rain_specific_attenuation(
        rainfall_rate, frequency
    )

    specific_attenuation = utils.transpose(specific_attenuation_array) * ma.ones(shape)

    two_way_attenuation = calc_two_way_attenuation(
        data.radar.height, specific_attenuation
    )

    two_way_attenuation[~inducing_region] = 0
    two_way_attenuation = ma.array(utils.ffill(two_way_attenuation.data))
    two_way_attenuation[two_way_attenuation == 0] = ma.masked

    return Attenuation(
        amount=two_way_attenuation,
        error=two_way_attenuation * 0.2,
        attenuated=affected_region,
        uncorrected=np.zeros_like(affected_region, dtype=bool),
    )


def _find_regions(
    classification: ClassificationResult,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Finds regions where rain attenuation is present and can be corrected or not."""
    warm_region = ~classification.category_bits.freezing
    is_rain = utils.transpose(classification.is_rain).astype(bool)
    affected_region = np.ones_like(warm_region, dtype=bool) * is_rain
    inducing_region = warm_region * is_rain
    return affected_region, inducing_region


def _calc_rain_specific_attenuation(
    rainfall_rate: np.ndarray, frequency: float
) -> np.ndarray:
    """Calculates specific attenuation due to rain (dB km-1).

    References:
        Crane, R. (1980). Prediction of Attenuation by Rain.
        IEEE Transactions on Communications, 28(9), 1717–1733.
        doi:10.1109/tcom.1980.1094844
    """
    if frequency > 8 and frequency < 12:
        alpha, beta = 0.0125, 1.18
    if frequency > 34 and frequency < 37:
        alpha, beta = 0.242, 1.04
    elif frequency > 93 and frequency < 96:
        alpha, beta = 0.95, 0.72
    else:
        msg = "Radar frequency not supported"
        raise ValueError(msg)
    return alpha * (rainfall_rate * con.M_S_TO_MM_H) ** beta
