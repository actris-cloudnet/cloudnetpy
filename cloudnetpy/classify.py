""" Classify gridded measurements. """

# import sys
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats
import droplet
import utils
from atmos import T0


def fetch_cat_bits(radar, beta, Tw, time, height, vfold):
    """ Classificate radar/lidar observations. """
    cat_bits = np.zeros(Tw.shape, dtype=int)
    if 'ldr' and 'v' not in radar:
        raise KeyError('Needs LDR and doppler velocity.')
    melting_bit = get_melting_bit_ldr(Tw, radar['ldr'], radar['v'])
    cold_bit = get_cold_bit(Tw, melting_bit, time, height)
    cloud_bit = droplet.get_liquid_layers(beta, height)
    rain_bit = get_rain_bit(radar['Zh'], time)
    clutter_bit = get_clutter_bit(rain_bit, radar['v'])
    insect_bit, insect_prob = get_insect_bit(radar, melting_bit, cloud_bit,
                                             rain_bit, clutter_bit, Tw, height)
    cat_bits = _set_cat_bits(cat_bits, cloud_bit, 1)
    cat_bits = _set_cat_bits(cat_bits, cold_bit, 3)
    cat_bits = _set_cat_bits(cat_bits, melting_bit, 4)
    cat_bits = _set_cat_bits(cat_bits, insect_bit, 6)
    return cat_bits


def get_melting_bit_ldr(Tw, ldr, v):
    """ Estimate melting layer from LDR data and model temperature.

    Args:
        Tw (ndarray): Wet bulb temperature, (n, m).
        ldr (ndarray): Linear depolarization ratio, (n, m).
        v (ndarray): Doppler velocity, (n, m).

    Returns:
        Binary field indicating the melting layer, (n, m) array
        where 1=yes and 0=no.

    """

    def _slice(arg1, arg2, ii, ind):
        out1, out2 = arg1[ii, ind], arg2[ii, ind]
        return out1, out2, ma.count(out1)

    def _basetop(dprof, pind, nind, a=10, b=2):
        top = droplet.get_top_ind(dprof, pind, nind, a, b)
        base = droplet.get_base_ind(dprof, pind, a, b)
        return top, base

    melting_bit = np.zeros(Tw.shape, dtype=int)
    ldr_diff = np.diff(ldr, axis=1).filled(fill_value=0)
    v_diff = np.diff(v, axis=1).filled(fill_value=0)
    trange = (-2, 5)  # find peak from this T range around T0
    for ii, tprof in enumerate(Tw):
        ind = np.where((tprof > T0+trange[0]) &
                       (tprof < T0+trange[1]))[0]
        nind = len(ind)
        ldr_prof, ldr_dprof, nldr = _slice(ldr, ldr_diff, ii, ind)
        v_prof, v_dprof, nv = _slice(v, v_diff, ii, ind)
        ldr_p = np.argmax(ldr_prof)
        v_p = np.argmax(v_dprof)
        if nldr > 3 or nv > 3:
            try:
                top, base = _basetop(ldr_dprof, ldr_p, nind)
                diff1 = ldr_prof[ldr_p] - ldr_prof[top]
                diff2 = ldr_prof[ldr_p] - ldr_prof[base]
                conds = (diff1 > 4,
                         diff2 > 4,
                         ldr_prof[ldr_p] > -20,
                         v_prof[base] < -2)
                if all(conds):
                    melting_bit[ii, ind[ldr_p]:ind[top]+1] = 1
            except:  # just cach all exceptions
                try:
                    top, base = _basetop(v_dprof, v_p, nind)
                    diff1 = v_prof[top] - v_prof[base]
                    if diff1 > 1 and v_prof[base] < -2:
                        melting_bit[ii, ind[v_p-1:v_p+2]] = 2
                except:  # failed whatever the reason
                    continue
    return melting_bit


def get_cold_bit(Tw, melting_bit, time, height):
    """ Find freezing region using the model temperature and melting layer.

    Sub-zero region is first derived from the model wet bulb temperature.
    It is then adjusted to start from the melting layer when we have such.
    Finally, a linear smoother is applied to combine the model and
    observations to avoid strong gradients in the zero-temperature line.

    Args:
        Tw (ndarray): Wet bulb temperature as (m, n) array.
        melting_bit (ndarray): Binary field indicating melting layer,
                               (m, n) array.
        time (ndarray): Time vector (m,).
        height (ndarray): Altitude vector (n,).

    Returns:
        Binary field indicating the sub-zero region, (m, n).

    Notes:
        It is not straightforward how the model temperature and melting
        layer should be combined to have a best possible estimate
        of the sub-zero region.

    """
    cold_bit = np.zeros(Tw.shape, dtype=int)
    ntime = time.shape[0]
    T0_alt = _get_T0_alt(Tw, height)
    mean_melting_height = np.zeros((ntime,))
    for ii in np.where(np.any(melting_bit, axis=1))[0]:
        mean_melting_height[ii] = np.median(
            height[np.where(melting_bit[ii, :])])
    m_final = np.copy(mean_melting_height)
    win = 240
    m_final[0] = mean_melting_height[0] or T0_alt[0]
    m_final[-1] = mean_melting_height[-1] or T0_alt[-1]
    for n in range(win, ntime-win):
        data_in_window = mean_melting_height[n-win:n+win+1]
        if not np.any(data_in_window):
            m_final[n] = T0_alt[n]
    ind = np.where(m_final > 0)[0]
    f = interp1d(time[ind], m_final[ind], kind='linear')
    tline = f(time)
    for ii, alt in enumerate(tline):
        cold_bit[ii, np.where(height > alt)[0]] = 1
    return cold_bit


def _set_cat_bits(cat_bits, bits_in, k):
    """ Update categorize-bits array. """
    ind = np.where(bits_in)
    cat_bits[ind] = utils.bit_set(cat_bits[ind], k)
    return cat_bits


def _get_T0_alt(Tw, height):
    """ Find altitudes where model temperature goes
        below freezing.

    Args:
        Tw (ndarray): Wet bulb temperature,
        height (ndarray): Altitude vector of the
                          wet bulb temperature.

    Returns:
        1D array containing the interpolated
        freezing altitudes.

    """
    alt = np.array([])
    for prof in Tw:
        ind = np.where(prof < T0)[0][0]
        if ind == 0:
            alt = np.append(alt, height[0])
        else:
            x = prof[ind-1:ind+1]
            y = height[ind-1:ind+1]
            x, y = zip(*sorted(zip(x, y)))
            alt = np.append(alt, np.interp(T0, x, y))
    return alt


def _insect_prob_ldr(z, ldr, z_loc=15, ldr_loc=-20):
    """ Probability that pixel is insect, based on Z and LDR values """
    zp, ldrp = np.zeros(z.shape), np.zeros(z.shape)
    ind = ~z.mask
    zp[ind] = stats.norm.cdf(z[ind]*-1, loc=z_loc, scale=8)
    ind = ~ldr.mask
    ldrp[ind] = stats.norm.cdf(ldr[ind], loc=ldr_loc, scale=5)
    return zp * ldrp


def _insect_prob_width(z, ldr, w, w_limit=0.06):
    """ (0, 1) Probability that pixel is insect, based on WIDTH values """
    i_prob = np.zeros(z.shape)
    temp_w = np.ones(z.shape)
    # pixels that have Z but no LDR
    ind = np.logical_and(ldr.mask, ~z.mask)
    temp_w[ind] = w[ind]
    i_prob[temp_w < w_limit] = 1
    return i_prob


def get_insect_bit(radar, melting_bit, droplet_bit, rain_bit,
                   clutter_bit, Tw, height, bit_lim=0.7):
    """Find insect probability from radar parameters.

    Args:
        radar (dict): Gridded radar fields that are (m, n).
        melting_bit (ndarray): Binary field for melting layer, (m, n).
        droplet_bit (ndarray): Binary field for liquid layers, (m, n).
        rain_bit (ndarray): Binary field for rainy profiles, (m,).
        clutter_bit (ndarra): Binary field for radar clutter, (m,).
        Tw (ndarray): Wet bulb temperature, (m, n).
        height (ndarray): Altitude vector, (n, ).
        bit_lim (float): Probability threshold between 0 and 1. Pixels where
            insect probability is greater that **bit_lim** are classified as 
            insects.

    Returns:
        Tuple containing insect_probability and insect binary flag (1=yes, 0=no).

    """
    insect_bit = np.zeros_like(clutter_bit)
    Z, ldr = radar['Zh'], radar['ldr']
    p1 = _insect_prob_ldr(Z, ldr)
    p2 = _insect_prob_width(Z, ldr, radar['width'])
    p_ins = p1 + p2
    p_ins[rain_bit == 1, :] = 0
    p_ins[droplet_bit == 1] = 0
    p_ins[melting_bit == 1] = 0
    p_ins[Tw < (T0-5)] = 0   # No insects below this temperature
    insect_bit[p_ins > bit_lim] = 1
    return insect_bit, p_ins


def get_rain_bit(Z, time, time_buffer=5):
    """ Find profiles affected by rain. 

    Args:
        Z (ndarray): Radar echo with shape (m, n).
        time (ndarray): Time vector with shape (m,).
        time_buffer (float, optional): If profile includes rain, 
            profiles measured **time_buffer** minutes before 
            and after are also flagged to contain rain. Defaults to 5.

    Returns:
        Binary array indicating profiles affected by rain (1=yes, 0=no).

    """    
    nprofs = len(time)
    rain_bit = np.zeros(nprofs, dtype=int)
    rain_bit[Z[:, 3] > 0] = 1
    step = utils.med_diff(time)*60*60  # minutes
    nsteps = int(round(time_buffer*60/step/2))
    for ind in np.where(rain_bit)[0]:
        i1 = max(0, ind-nsteps)
        i2 = min(ind+nsteps+1, nprofs)
        rain_bit[i1:i2] = 1
    return rain_bit


def get_clutter_bit(rain_bit, v, ngates=10, vlim=0.05):
    """ Estimate clutter from radar data. """
    clutter_bit = np.zeros(v.shape, dtype=int)
    no_rain = np.where(rain_bit == 0)[0]
    ind = np.ma.where(np.abs(v[no_rain, 0:ngates]) < vlim)
    for n, m in zip(*ind):
        clutter_bit[no_rain[n], m] = 1
    return clutter_bit
