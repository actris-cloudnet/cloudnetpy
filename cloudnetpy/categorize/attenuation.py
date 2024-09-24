from numpy import ma

from cloudnetpy.categorize.attenuations import (
    RadarAttenuation,
    gas_attenuation,
    liquid_attenuation,
    melting_attenuation,
    rain_attenuation,
)
from cloudnetpy.categorize.containers import ClassificationResult, Observations


def get_attenuations(
    data: Observations, classification: ClassificationResult
) -> RadarAttenuation:
    rain = rain_attenuation.calc_rain_attenuation(data, classification)
    gas = gas_attenuation.calc_gas_attenuation(data, classification)
    liquid = liquid_attenuation.LiquidAttenuation(data, classification).attenuation
    melting = melting_attenuation.calc_melting_attenuation(data, classification)

    liquid.amount[rain.attenuated] = ma.masked
    liquid.error[rain.attenuated] = ma.masked
    liquid.attenuated[rain.attenuated] = False
    liquid.uncorrected[rain.attenuated] = False

    return RadarAttenuation(
        gas=gas,
        liquid=liquid,
        rain=rain,
        melting=melting,
    )
