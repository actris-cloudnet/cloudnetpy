"""Constants used in Cloudnet processing."""
from typing import Final

# Triple point of water
T0: Final = 273.16

# Ratio of the molecular weight of water vapor to dry air
MW_RATIO: Final = 0.62198

# Specific heat capacity of air at around 275K (J kg-1 K-1)
SPECIFIC_HEAT: Final = 1004

# Latent heat of evaporation (J kg-1)
LATENT_HEAT: Final = 2.26e6

# Specific gas constant for dry  air (J kg-1 K-1)
RS: Final = 287.058

# ice density kg m-3
RHO_ICE: Final = 917

# other
SEC_IN_MINUTE: Final = 60
SEC_IN_HOUR: Final = 3600
SEC_IN_DAY: Final = 86400
MM_TO_M: Final = 1e-3
G_TO_KG: Final = 1e-3
