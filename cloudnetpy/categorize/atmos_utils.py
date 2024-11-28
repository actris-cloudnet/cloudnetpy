import logging

import numpy as np
import numpy.typing as npt
import scipy.constants
from numpy import ma

import cloudnetpy.constants as con
from cloudnetpy import utils


def calc_wet_bulb_temperature(model_data: dict) -> np.ndarray:
    """Calculate wet-bulb temperature iteratively.

    Args:
        model_data: Model variables `temperature`, `pressure`, `q`.

    Returns:
        Wet-bulb temperature (K).

    References:
        Al-Ismaili, A. M., & Al-Azri, N. A. (2016). Simple Iterative Approach to
        Calculate Wet-Bulb Temperature for Estimating Evaporative Cooling
        Efficiency. Int. J. Agric. Innovations Res., 4, 1013-1018.
    """
    specific_humidity = model_data["q"]
    pressure = model_data["pressure"]
    td = k2c(model_data["temperature"])
    vp = calc_vapor_pressure(pressure, specific_humidity)
    W = calc_mixing_ratio(vp, pressure)
    L_v_0 = 2501e3  # Latent heat of vaporization at 0degC (J kg-1)

    def f(tw):
        svp = calc_saturation_vapor_pressure(c2k(tw))
        W_s = calc_mixing_ratio(svp, pressure)
        C_p_w = 0.0265 * tw**2 - 1.7688 * tw + 4205.6  # Eq. 6 (J kg-1 C-1)
        C_p_wv = 0.0016 * td**2 + 0.1546 * td + 1858.7  # Eq. 7 (J kg-1 C-1)
        C_p_da = 0.0667 * ((td + tw) / 2) + 1005  # Eq. 8 (J kg-1 C-1)
        a = (L_v_0 - (C_p_w - C_p_wv) * tw) * W_s - C_p_da * (td - tw)
        b = L_v_0 + C_p_wv * td - C_p_w * tw
        return a / b - W

    min_err = 1e-6 * np.maximum(np.abs(td), 1)
    delta = 1e-8
    tw = td
    max_iter = 20
    for _ in range(max_iter):
        f_tw = f(tw)
        if np.all(np.abs(f_tw) < min_err):
            break
        df_tw = (f(tw + delta) - f_tw) / delta
        tw = tw - f_tw / df_tw
    else:
        msg = (
            "Wet-bulb temperature didn't converge after %d iterations: "
            "error min %g, max %g, mean %g, median %g"
        )
        logging.warning(
            msg, max_iter, np.min(f_tw), np.max(f_tw), np.mean(f_tw), np.median(f_tw)
        )

    return c2k(tw)


def calc_vapor_pressure(
    pressure: npt.NDArray, specific_humidity: npt.NDArray
) -> npt.NDArray:
    """Calculate vapor pressure of water based on pressure and specific
    humidity.

    Args:
        pressure: Pressure (Pa)
        specific_humidity: Specific humidity (1)

    Returns:
        Vapor pressure (Pa)

    References:
        Cai, J. (2019). Humidity Measures.
        https://cran.r-project.org/web/packages/humidity/vignettes/humidity-measures.html
    """
    return (
        specific_humidity
        * pressure
        / (con.MW_RATIO + (1 - con.MW_RATIO) * specific_humidity)
    )


def c2k(temp: np.ndarray) -> np.ndarray:
    """Converts Celsius to Kelvins."""
    return ma.array(temp) + 273.15


def k2c(temp: np.ndarray) -> np.ndarray:
    """Converts Kelvins to Celsius."""
    return ma.array(temp) - 273.15


def find_cloud_bases(array: np.ndarray) -> np.ndarray:
    """Finds bases of clouds.

    Args:
        array: 2D boolean array denoting clouds or some other similar field.

    Returns:
        Boolean array indicating bases of the individual clouds.

    """
    zeros = np.zeros(array.shape[0])
    array_padded = np.insert(array, 0, zeros, axis=1).astype(int)
    return np.diff(array_padded, axis=1) == 1


def find_cloud_tops(array: np.ndarray) -> np.ndarray:
    """Finds tops of clouds.

    Args:
        array: 2D boolean array denoting clouds or some other similar field.

    Returns:
        Boolean array indicating tops of the individual clouds.

    """
    array_flipped = np.fliplr(array)
    bases_of_flipped = find_cloud_bases(array_flipped)
    return np.fliplr(bases_of_flipped)


def find_lowest_cloud_bases(
    cloud_mask: np.ndarray,
    height: np.ndarray,
) -> ma.MaskedArray:
    """Finds altitudes of cloud bases."""
    cloud_heights = cloud_mask * height
    return _find_lowest_heights(cloud_heights)


def find_highest_cloud_tops(
    cloud_mask: np.ndarray,
    height: np.ndarray,
) -> ma.MaskedArray:
    """Finds altitudes of cloud tops."""
    cloud_heights = cloud_mask * height
    cloud_heights_flipped = np.fliplr(cloud_heights)
    return _find_lowest_heights(cloud_heights_flipped)


def _find_lowest_heights(cloud_heights: np.ndarray) -> ma.MaskedArray:
    inds = (cloud_heights != 0).argmax(axis=1)
    heights = np.array([cloud_heights[i, ind] for i, ind in enumerate(inds)])
    return ma.masked_equal(heights, 0.0)


def fill_clouds_with_lwc_dz(
    temperature: np.ndarray, pressure: np.ndarray, is_liquid: np.ndarray
) -> np.ndarray:
    """Fills liquid clouds with lwc change rate at the cloud bases.

    Args:
        temperature: 2D temperature array (K).
        pressure: 2D pressure array (Pa).
        is_liquid: Boolean array indicating presence of liquid clouds.

    Returns:
        Liquid water content change rate (kg m-3 m-1), so that for each cloud the base
        value is filled for the whole cloud.

    """
    lwc_dz = get_lwc_change_rate_at_bases(temperature, pressure, is_liquid)
    lwc_dz_filled = ma.zeros(lwc_dz.shape)
    lwc_dz_filled[is_liquid] = utils.ffill(lwc_dz[is_liquid])
    return lwc_dz_filled


def get_lwc_change_rate_at_bases(
    temperature: np.ndarray,
    pressure: np.ndarray,
    is_liquid: np.ndarray,
) -> np.ndarray:
    """Finds LWC change rate in liquid cloud bases.

    Args:
        temperature: 2D temperature array (K).
        pressure: 2D pressure array (Pa).
        is_liquid: Boolean array indicating presence of liquid clouds.

    Returns:
        Liquid water content change rate at cloud bases (kg m-3 m-1).

    """
    liquid_bases = find_cloud_bases(is_liquid)
    lwc_dz = ma.zeros(liquid_bases.shape)
    lwc_dz[liquid_bases] = calc_lwc_change_rate(
        temperature[liquid_bases],
        pressure[liquid_bases],
    )

    return lwc_dz


def calc_lwc_change_rate(temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Returns rate of change of condensable water (LWC).

    Calculates the theoretical adiabatic rate of increase of LWC
    with height, given the cloud base temperature and pressure.

    Args:
        temperature: Temperature of cloud base (K).
        pressure: Pressure of cloud base (Pa).

    Returns:
        dlwc/dz (kg m-3 m-1)

    References:
        Brenguier, 1991, https://doi.org/10.1175/1520-0469(1991)048<0264:POTCPA>2.0.CO;2

    """
    svp = calc_saturation_vapor_pressure(temperature)
    svp_mixing_ratio = calc_mixing_ratio(svp, pressure)
    air_density = calc_air_density(pressure, temperature, svp_mixing_ratio)

    e = 0.622
    Cp = 1004  # J kg-1 K-1
    Lv = 2.45e6  # J kg-1 = Pa m3 kg-1
    qs = svp_mixing_ratio  # kg kg-1
    pa = air_density  # kg m-3
    es = svp  # Pa
    P = pressure  # Pa
    T = temperature  # K

    # See Appendix B in Brenguier (1991) for the derivation of the following equation
    dqs_dp = (
        -(1 - (Cp * T) / (e * Lv))
        * (((Cp * T) / (e * Lv)) + ((Lv * qs * pa) / (P - es))) ** -1
        * (e * es)
        * (P - es) ** -2
    )

    # Using hydrostatic equation to convert dqs_dp to dqs_dz
    dqs_dz = dqs_dp * air_density * -scipy.constants.g

    return dqs_dz * air_density


def calc_saturation_vapor_pressure(temperature: np.ndarray) -> np.ndarray:
    """Goff-Gratch formula for saturation vapor pressure over water adopted by WMO.

    Args:
        temperature: Temperature (K).

    Returns:
        Saturation vapor pressure (Pa).

    """
    ratio = con.T0 / temperature
    inv_ratio = ratio**-1
    return (
        10
        ** (
            10.79574 * (1 - ratio)
            - 5.028 * np.log10(inv_ratio)
            + 1.50475e-4 * (1 - (10 ** (-8.2969 * (inv_ratio - 1))))
            + 0.42873e-3 * (10 ** (4.76955 * (1 - ratio)) - 1)
            + 0.78614
        )
    ) * con.HPA_TO_PA


def calc_mixing_ratio(vapor_pressure: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    """Calculates mixing ratio from partial vapor pressure and pressure.

    Args:
        vapor_pressure: Partial pressure of water vapor (Pa).
        pressure: Atmospheric pressure (Pa).

    Returns:
        Mixing ratio (kg kg-1).

    """
    return con.MW_RATIO * vapor_pressure / (pressure - vapor_pressure)


def calc_air_density(
    pressure: np.ndarray,
    temperature: np.ndarray,
    svp_mixing_ratio: np.ndarray,
) -> np.ndarray:
    """Calculates air density (kg m-3).

    Args:
        pressure: Pressure (Pa).
        temperature: Temperature (K).
        svp_mixing_ratio: Saturation vapor pressure mixing ratio (kg kg-1).

    Returns:
        Air density (kg m-3).

    """
    return pressure / (con.RS * temperature * (0.6 * svp_mixing_ratio + 1))


def calc_adiabatic_lwc(lwc_dz: np.ndarray, height: np.ndarray) -> np.ndarray:
    """Calculates adiabatic liquid water content (kg m-3).

    Args:
        lwc_dz: Liquid water content change rate (kg m-3 m-1) calculated at the
            base of each cloud and filled to that cloud.
        height: Height vector (m).

    Returns:
        Liquid water content (kg m-3).

    """
    is_cloud = lwc_dz != 0
    cloud_indices = utils.cumsumr(is_cloud, axis=1)
    dz = utils.path_lengths_from_ground(height) * np.ones_like(lwc_dz)
    dz[cloud_indices < 1] = 0
    return utils.cumsumr(dz, axis=1) * lwc_dz


def normalize_lwc_by_lwp(
    lwc_adiabatic: np.ndarray, lwp: np.ndarray, height: np.ndarray
) -> np.ndarray:
    """Finds LWC that would produce measured LWP.

    Calculates LWP-weighted, normalized LWC. This is the measured
    LWP distributed to liquid cloud pixels according to their
    theoretical proportion.

    Args:
        lwc_adiabatic: Theoretical 2D liquid water content (kg m-3).
        lwp: 1D liquid water path (kg m-2).
        height: Height vector (m).

    Returns:
        2D LWP-weighted, scaled LWC (kg m-3) that would produce the observed LWP.

    """
    path_lengths = utils.path_lengths_from_ground(height)
    theoretical_lwp = ma.sum(lwc_adiabatic * path_lengths, axis=1)
    scaling_factors = lwp / theoretical_lwp
    return lwc_adiabatic * utils.transpose(scaling_factors)


def calc_altitude(temperature: float, pressure: float) -> float:
    """Calculate altitude (m) based on observed pressure (Pa) and temperature (K)
    using the International Standard Atmosphere (ISA) model.

    Args:
        temperature: Observed temperature (K).
        pressure: Observed atmospheric pressure (Pa).

    Returns:
        Altitude (m).
    """
    L = 0.0065  # Temperature lapse rate (K/m)
    return (temperature / L) * (1 - (pressure / con.P0) ** (con.RS * L / con.G))
