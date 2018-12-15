""" Classify gridded measurements. """

# import sys
import numpy as np
import numpy.ma as ma
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import droplet
import utils
from atmos import T0


def fetch_cat_bits(radar, lidar, model, time, height, vfold):
    """ Experimental classification based on lidar and LDR-supported
    radar data. """
    cat_bits = np.zeros(model['Tw'].shape, dtype=int)
    if 'ldr' and 'v' not in radar:
        raise KeyError('Needs LDR and doppler velocity.')
    melting_bit = get_melting_bit_ldr(model['Tw'], radar['ldr'], radar['v'])
    cold_bit = get_cold_bit(model['Tw'], melting_bit, time, height)
    cloud_bit = droplet.get_liquid_layers(lidar['beta'], height)

    cat_bits = _set_cat_bits(cat_bits, cloud_bit, 1)
    cat_bits = _set_cat_bits(cat_bits, cold_bit, 3)
    cat_bits = _set_cat_bits(cat_bits, melting_bit, 4)

    return cat_bits


def get_melting_bit_ldr(Tw, ldr, v):
    """ Estimate melting layer from LDR data and model temperature
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
        Tw (array_like): Wet bulb temperature as (m, n) array.
        melting_bit (array_like): Binary field indicating melting layer,
        (m, n) array.
        time (array_like): Time vector (m,).
        height (array_like): Altitude vector (n,).

    Returns:
        Binary field for melting layer region, (m, n) array.

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
        Tw (array_like): Wet bulb temperature,
        height (array_like): Altitude vector of the
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
