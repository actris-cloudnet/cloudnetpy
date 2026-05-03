import atmoslib
import numpy as np
import numpy.typing as npt
from numpy import ma

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

    specific_attenuation_array = atmoslib.rain_specific_attenuation(
        rainfall_rate * con.M_S_TO_MM_H, frequency
    )

    specific_attenuation = utils.transpose(specific_attenuation_array) * ma.ones(shape)

    two_way_attenuation = calc_two_way_attenuation(
        data.radar.height_agl, specific_attenuation
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
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """Finds regions where rain attenuation is present and can be corrected or not."""
    warm_region = ~classification.category_bits.freezing
    is_rain = utils.transpose(classification.is_rain).astype(bool)
    affected_region = np.ones_like(warm_region, dtype=bool) * is_rain
    inducing_region = warm_region * is_rain
    return affected_region, inducing_region
