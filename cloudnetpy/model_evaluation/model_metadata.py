"""Model-agnostic configuration for the model evaluation subpackage.

All model files consumed here are harmonized by the model munger to a common
set of variable names and units (temperature [K], pressure [Pa], qi/ql/
cloud_fraction [1], uwind/vwind [m s-1], height [m]). There is therefore no
per-model variable mapping: the only thing that differs between models is the
number of vertical levels, which is handled dynamically via `ALTITUDE_LIMIT`.
"""

# Mapping from internal product keys to the harmonized model-file variable
# names. These names are written identically for every model by the munger.
MODEL_VARIABLE_NAMES = {
    "T": "temperature",
    "p": "pressure",
    "h": "height",
    "iwc": "qi",
    "lwc": "ql",
    "cf": "cloud_fraction",
}

# Coordinate / metadata variables that are identical between forecast cycles.
COMMON_VARIABLES = (
    "time",
    "level",
    "latitude",
    "longitude",
    "horizontal_resolution",
)

# Variables that may differ between forecast cycles.
CYCLE_VARIABLES = ("forecast_time", "height")

# Vertical levels above this altitude (m) are dropped: the radar cannot observe
# them and the model grid above is not evaluated.
ALTITUDE_LIMIT = 22000.0
