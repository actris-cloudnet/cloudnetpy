""" This module contains some hand-tuned metadata and 
ad-hoc parameter values (that might be site/instrument
dependent in future.)
"""

# Institute running the software.
INSTITUTE = 'Finnish Meteorological Institute'

# Fractional and linear error components
# for liquid water path (LWP)
LWP_ERROR = (0.25, 20)

# Estimated precision of gas attenuation
GAS_ATTEN_PREC = 0.1

# Random error and bias in lidar backscattering
BETA_ERROR = (0.5, 3.0)

# Estimate of the bias in cloud radar echo
Z_BIAS = 1
