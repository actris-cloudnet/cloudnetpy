"""Constants used in Cloudnet processing."""

import math
from typing import Final

T_FREEZING: Final = 273.15
"""Freezing point of water at standard pressure (K), i.e. 0 degrees Celsius"""

RHO_ICE: Final = 917
"""Ice density (kg m-3)"""

RHO_WATER: Final = 1000
"""Liquid water density (kg m-3)"""

SPEED_OF_LIGHT: Final = 299_792_458
"""Speed of light in vacuum (m s-1)"""

# Unit conversions
SEC_IN_MINUTE: Final = 60
SEC_IN_HOUR: Final = 3600
SEC_IN_DAY: Final = 86400
MM_TO_M: Final = 1e-3
MM2_TO_M2: Final = 1e-6
G_TO_KG: Final = 1e-3
M_TO_KM: Final = 1e-3
KM_TO_M: Final = 1e3
KG_TO_G: Final = 1e3
M_TO_MM: Final = 1e3
CM_TO_M: Final = 1e-2
M_S_TO_MM_H: Final = SEC_IN_HOUR / MM_TO_M
MM_H_TO_M_S: Final = 1 / M_S_TO_MM_H
GHZ_TO_HZ: Final = 1e9
HZ_TO_GHZ: Final = 1e-9
HPA_TO_PA: Final = 100
KM_H_TO_M_S: Final = 1000 / SEC_IN_HOUR

CM_TO_KG_M2: Final = 10
"""Liquid water equivalent cm to kg m-2"""

LN_TO_DB: Final = 10 / math.log(10)
"""Natural logarithm of a power ratio to decibels"""
