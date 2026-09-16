"""Constants used in Cloudnet processing."""

import math
from typing import Final

# Triple point of water
T0: Final = 273.16

# Freezing point of water at standard pressure (K), i.e. 0 degrees Celsius
T_FREEZING: Final = 273.15

# Specific gas constant for dry  air (J kg-1 K-1)
RS: Final = 287.058

# ice density kg m-3
RHO_ICE: Final = 917

# liquid water density kg m-3
RHO_WATER: Final = 1000

# other
SPEED_OF_LIGHT: Final = 299_792_458
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
TWO_WAY: Final = 2
G: Final = 9.80665

# Natural logarithm of a power ratio to decibels
LN_TO_DB: Final = 10 / math.log(10)
