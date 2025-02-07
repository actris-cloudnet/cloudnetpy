"""Constants used in Cloudnet processing."""

from typing import Final

# Triple point of water
T0: Final = 273.16

# Ratio of the molecular weight of water vapor to dry air
MW_RATIO: Final = 0.62198

# Specific gas constant for dry  air (J kg-1 K-1)
RS: Final = 287.058

# ice density kg m-3
RHO_ICE: Final = 917

# Standard atmospheric pressure at sea level Pa
P0: Final = 1013_25

# other
SPEED_OF_LIGHT: Final = 3.0e8
SEC_IN_MINUTE: Final = 60
SEC_IN_HOUR: Final = 3600
SEC_IN_DAY: Final = 86400
MM_TO_M: Final = 1e-3
G_TO_KG: Final = 1e-3
M_TO_KM: Final = 1e-3
KG_TO_G: Final = 1e3
M_TO_MM: Final = 1e3
M_S_TO_MM_H: Final = SEC_IN_HOUR / MM_TO_M
MM_H_TO_M_S: Final = 1 / M_S_TO_MM_H
GHZ_TO_HZ: Final = 1e9
HPA_TO_PA: Final = 100
PA_TO_HPA: Final = 1 / HPA_TO_PA
KM_H_TO_M_S: Final = 1000 / SEC_IN_HOUR
TWO_WAY: Final = 2
G: Final = 9.80665
