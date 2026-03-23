import numpy as np
import numpy.typing as npt
from numpy import ma
from scipy import ndimage

from cloudnetpy import utils
from cloudnetpy.categorize import (
    atmos_utils,
    droplet,
    falling,
    freezing,
    insects,
    melting,
)
from cloudnetpy.categorize.attenuations import RadarAttenuation
from cloudnetpy.categorize.containers import (
    ClassData,
    ClassificationResult,
    Observations,
)
from cloudnetpy.constants import T0
from cloudnetpy.products.product_tools import CategoryBits, QualityBits


def classify_measurements(data: Observations) -> ClassificationResult:
    """Classifies radar/lidar observations.

    This function classifies atmospheric scatterers from the input data.
    The input data needs to be averaged or interpolated to the common
    time / height grid before calling this function.

    Args:
        data: A :class:`Observations` instance.

    Returns:
        A :class:`ClassificationResult` instance.

    References:
        The Cloudnet classification scheme is based on methodology proposed by
        Hogan R. and O'Connor E., 2004, https://bit.ly/2Yjz9DZ and its
        proprietary Matlab implementation.

    Notes:
        Some individual classification methods are changed in this Python
        implementation compared to the original Cloudnet methodology.
        Especially methods classifying insects, melting layer and liquid droplets.

    """
    bits = CategoryBits(
        droplet=np.array([], dtype=bool),
        falling=np.array([], dtype=bool),
        freezing=np.array([], dtype=bool),
        melting=np.array([], dtype=bool),
        aerosol=np.array([], dtype=bool),
        insect=np.array([], dtype=bool),
    )

    obs = ClassData(data)

    bits.melting = melting.find_melting_layer(obs)
    bits.freezing = freezing.find_freezing_region(obs, bits.melting)
    liquid_from_lidar = droplet.find_liquid(obs)
    if obs.lv0_files is not None and len(obs.lv0_files) > 0:
        if "rpg-fmcw-94" not in obs.radar_type.lower():
            msg = "VoodooNet is only implemented for RPG-FMCW-94 radar."
            raise NotImplementedError(msg)
        import voodoonet  # noqa: PLC0415

        options = voodoonet.VoodooOptions(progress_bar=False)
        dumb_date = obs.date.isoformat().split("-")
        target_time = voodoonet.utils.decimal_hour2unix(dumb_date, obs.time)
        liquid_prob = voodoonet.infer(
            list(obs.lv0_files), target_time=target_time, options=options
        )
        liquid_from_radar = liquid_prob > 0.55
        liquid_from_radar = _remove_false_radar_liquid(
            liquid_from_radar,
            liquid_from_lidar,
        )
        liquid_from_radar[~bits.freezing] = 0
        is_liquid = liquid_from_radar | liquid_from_lidar
    else:
        is_liquid = liquid_from_lidar
        liquid_prob = None
    bits.droplet = droplet.correct_liquid_top(obs, is_liquid, bits.freezing, limit=500)
    bits.insect, insect_prob = insects.find_insects(obs, bits.melting, bits.droplet)
    bits.falling = falling.find_falling_hydrometeors(obs, bits.droplet, bits.insect)
    for _ in range(5):
        _fix_undetected_melting_layer(bits)
        _filter_insects(bits)
    bits.aerosol = _find_aerosols(obs, bits)
    _fix_super_cold_liquid(obs, bits)
    _reclassify_ice_with_high_ldr(obs, bits)
    _filter_insects_in_falling(bits)
    _filter_falling(bits, obs)

    is_clutter = _remove_clutter_in_liquid(obs.is_clutter, bits)

    return ClassificationResult(
        category_bits=bits,
        is_rain=obs.is_rain,
        is_clutter=is_clutter,
        insect_prob=insect_prob,
        liquid_prob=liquid_prob,
    )


def fetch_quality(
    data: Observations,
    classification: ClassificationResult,
    attenuations: RadarAttenuation,
) -> QualityBits:
    return QualityBits(
        radar=~data.radar.data["Z"][:].mask,
        lidar=~data.lidar.data["beta"][:].mask,
        clutter=classification.is_clutter,
        molecular=np.zeros(data.radar.data["Z"][:].shape, dtype=bool),
        attenuated_liquid=attenuations.liquid.attenuated,
        corrected_liquid=attenuations.liquid.attenuated
        & ~attenuations.liquid.uncorrected,
        attenuated_rain=attenuations.rain.attenuated,
        corrected_rain=attenuations.rain.attenuated & ~attenuations.rain.uncorrected,
        attenuated_melting=attenuations.melting.attenuated,
        corrected_melting=attenuations.melting.attenuated
        & ~attenuations.melting.uncorrected,
    )


def _reclassify_ice_with_high_ldr(obs: ClassData, bits: CategoryBits) -> None:
    """Reclassifies ice pixels with high LDR as insects.

    Ice particles typically have LDR below -13 dB. Higher values in the
    freezing region (excluding the melting layer) indicate insects, which
    can be present in temperatures down to about -10 C.
    """
    if not hasattr(obs, "ldr"):
        return
    ldr_limit = -13
    temp_limit = T0 - 10
    is_ice = bits.falling & bits.freezing & ~bits.droplet & ~bits.melting
    high_ldr = ~ma.getmaskarray(obs.ldr) & (obs.ldr > ldr_limit)
    warm_enough = obs.tw > temp_limit
    above_melting = utils.ffill(bits.melting)
    reclassify = is_ice & high_ldr & warm_enough & ~above_melting
    bits.insect[reclassify] = True
    bits.falling[reclassify] = False
    has_lidar = ~obs.beta.mask
    bits.aerosol[reclassify & has_lidar] = True


def _filter_insects_in_falling(
    bits: CategoryBits,
    min_falling_size: int = 100,
    dilation: int = 2,
) -> None:
    """Reclassifies insect pixels in freezing regions near large falling ice.

    Insect pixels in freezing temperatures near sufficiently large falling
    hydrometeor regions are almost certainly false positives. Real insects
    cannot survive in freezing conditions, so the freezing constraint
    protects legitimate warm-region insects from being reclassified.

    Args:
        bits: A :class:`CategoryBits` instance.
        min_falling_size: Minimum size of nearby falling regions required
            to trigger reclassification.
        dilation: Number of pixels to dilate around falling regions.
    """
    structure = ndimage.generate_binary_structure(2, 1)
    falling_labels, n_falling = ndimage.label(bits.falling, structure=structure)
    if n_falling == 0:
        return
    falling_ids = np.arange(1, n_falling + 1)
    falling_sizes = ndimage.sum(bits.falling, falling_labels, falling_ids)
    large_falling = np.isin(
        falling_labels, falling_ids[falling_sizes >= min_falling_size]
    )
    near_large_falling = ndimage.binary_dilation(
        large_falling, structure, iterations=dilation
    )
    reclassify = bits.insect & bits.freezing & near_large_falling
    bits.falling[reclassify] = True
    bits.insect[reclassify] = False
    bits.aerosol[reclassify] = False


def _fix_super_cold_liquid(obs: ClassData, bits: CategoryBits) -> None:
    """Supercooled liquid droplets do not exist in atmosphere below around -38 C."""
    t_limit = T0 - 38
    super_cold_liquid = np.where((obs.tw < t_limit) & bits.droplet)
    bits.droplet[super_cold_liquid] = False
    bits.falling[super_cold_liquid] = True


def _remove_false_radar_liquid(
    liquid_from_radar: npt.NDArray,
    liquid_from_lidar: npt.NDArray,
) -> npt.NDArray[np.bool_]:
    """Removes radar-liquid below lidar-detected liquid bases."""
    lidar_liquid_bases = atmos_utils.find_cloud_bases(liquid_from_lidar)
    for prof, base in zip(*np.where(lidar_liquid_bases), strict=True):
        liquid_from_radar[prof, 0:base] = 0
    return liquid_from_radar


def _find_aerosols(
    obs: ClassData,
    bits: CategoryBits,
) -> npt.NDArray[np.bool_]:
    """Estimates aerosols from lidar backscattering.

    Aerosols are lidar signals that are: a) not falling, b) not liquid droplets.

    Args:
        obs: A :class:`ClassData` instance.
        bits: A :class:`CategoryBits instance.

    Returns:
        2-D boolean array containing aerosols.

    """
    is_beta = ~obs.beta.mask
    return is_beta & ~bits.falling & ~bits.droplet


def _fix_undetected_melting_layer(bits: CategoryBits) -> None:
    drizzle_and_falling = _find_drizzle_and_falling(bits)
    transition = ma.diff(drizzle_and_falling, axis=1) == -1
    bits.melting[:, 1:][transition] = True


def _find_drizzle_and_falling(bits: CategoryBits) -> npt.NDArray:
    """Classifies pixels as falling, drizzle and others.

    Args:
        bits: A :class:`CategoryBits instance.

    Returns:
        2D array where values are 1 (falling, drizzle, supercooled liquids),
        2 (drizzle), and masked (all others).

    """
    falling_dry = bits.falling & ~bits.droplet
    supercooled_liquids = bits.droplet & bits.freezing
    drizzle = falling_dry & ~bits.freezing
    drizzle_and_falling = falling_dry.astype(int) + drizzle.astype(int)
    drizzle_and_falling = ma.copy(drizzle_and_falling)
    drizzle_and_falling[supercooled_liquids] = 1
    drizzle_and_falling[drizzle_and_falling == 0] = ma.masked
    return drizzle_and_falling


def _filter_insects(bits: CategoryBits) -> None:
    is_melting_layer = bits.melting
    is_insects = bits.insect
    is_falling = bits.falling

    # Remove above melting layer
    above_melting = utils.ffill(is_melting_layer)
    ind = np.where(is_insects & above_melting)
    is_falling[ind] = True
    is_insects[ind] = False

    # remove around melting layer:
    original_insects = np.copy(is_insects)
    n_gates = 5
    for x, y in zip(*np.where(is_melting_layer), strict=True):
        try:
            # change insects to drizzle below melting layer pixel
            ind1 = np.arange(y - n_gates, y)
            ind11 = np.where(original_insects[x, ind1])[0]
            n_drizzle = sum(is_falling[x, :y])
            if n_drizzle > 5:
                is_falling[x, ind1[ind11]] = True
                is_insects[x, ind1[ind11]] = False
            else:
                continue
            # change insects on the right and left of melting layer pixel to drizzle
            ind1 = np.arange(x - n_gates, x + n_gates + 1)
            ind11 = np.where(original_insects[ind1, y])[0]
            is_falling[ind1[ind11], y - 1 : y + 2] = True
            is_insects[ind1[ind11], y - 1 : y + 2] = False
        except IndexError:
            continue
    bits.falling = is_falling
    bits.insect = is_insects


def _remove_clutter_in_liquid(
    is_clutter: npt.NDArray,
    bits: CategoryBits,
) -> npt.NDArray:
    """Removes clutter that is inside liquid or falling hydrometeor layers.

    Clutter detection based on near-zero velocity can produce false positives
    inside cloud and drizzle layers. This function removes clutter flags from
    pixels that are adjacent to droplet or falling hydrometeor pixels, and
    reclassifies them as falling hydrometeors.
    """
    is_clutter = is_clutter.copy()
    liquid_or_falling = bits.droplet | bits.falling
    structure = ndimage.generate_binary_structure(2, 1)
    near_liquid = ndimage.binary_dilation(liquid_or_falling, structure)
    false_clutter = is_clutter & near_liquid
    is_clutter[false_clutter] = False
    bits.falling[false_clutter] = True
    return is_clutter


def _filter_falling(bits: CategoryBits, obs: ClassData) -> None:
    # filter falling ice speckle noise
    is_beta = ~obs.beta.mask
    is_freezing = bits.freezing
    is_falling = bits.falling
    filtered_out = is_falling & ~np.asarray(
        utils.remove_small_objects(
            is_falling,
            max_size=10,
            connectivity=1,
        ),
        dtype=bool,
    )
    is_falling[filtered_out] = False
    # In warm conditions, these are likely insects
    bits.insect[filtered_out & ~is_freezing] = True
    bits.aerosol[filtered_out & ~is_freezing & is_beta] = True
    # In cold conditions, classify as aerosol if lidar signal is present
    bits.aerosol[filtered_out & is_freezing & is_beta] = True
    bits.falling = is_falling
