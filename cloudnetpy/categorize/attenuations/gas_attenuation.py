import numpy as np

from cloudnetpy.categorize.attenuations import (
    Attenuation,
    calc_two_way_attenuation,
)
from cloudnetpy.categorize.containers import ClassificationResult, Observations


def calc_gas_attenuation(
    data: Observations, classification: ClassificationResult
) -> Attenuation:
    model_data = data.model.data_dense

    specific_attenuation = model_data["specific_gas_atten"].copy()
    saturated_attenuation = model_data["specific_saturated_gas_atten"]

    liquid_in_pixel = classification.category_bits.droplet
    specific_attenuation[liquid_in_pixel] = saturated_attenuation[liquid_in_pixel]

    two_way_attenuation = calc_two_way_attenuation(
        data.radar.height, specific_attenuation
    )

    return Attenuation(
        amount=two_way_attenuation,
        error=two_way_attenuation * 0.1,
        attenuated=np.ones_like(two_way_attenuation, dtype=bool),
        uncorrected=np.zeros_like(two_way_attenuation, dtype=bool),
    )
