""" Classify gridded measurements. """

# import sys
import numpy as np
import numpy.ma as ma
# import matplotlib as mpl
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import stats
from cloudnetpy import droplet
from cloudnetpy import utils
from cloudnetpy.atmos import T0


def fetch_cat_bits(radar, beta, Tw, time, height):
    """Classifies radar/lidar observations.

    Args:
        radar: A dict containing gridded radar fields
            ('Zh', 'v', 'ldr', 'width').
        beta (ndarray): Attenuated backscattering coefficient.
        Tw (ndarray): Wet bulb temperature.
        height (ndarray): 1D altitude vector.

    Returns:
        Bit field containing the classification.

    """
    cat_bits = np.zeros(Tw.shape, dtype=int)
    if 'ldr' and 'v' not in radar:
        raise KeyError('Needs LDR and doppler velocity.')
    melting_bit = get_melting_bit_ldr(Tw, radar['ldr'], radar['v'])
    cold_bit = get_cold_bit(Tw, melting_bit, time, height)
    cloud_bit = droplet.get_liquid_layers(beta, height)
    rain_bit = get_rain_bit(radar['Zh'], time)
    clutter_bit = get_clutter_bit(radar['v'], rain_bit)
    insect_bit, insect_prob = get_insect_bit(radar, Tw, melting_bit, cloud_bit,
                                             rain_bit, clutter_bit)
    cat_bits = _set_cat_bits(cat_bits, cloud_bit, 1)
    cat_bits = _set_cat_bits(cat_bits, cold_bit, 3)
    cat_bits = _set_cat_bits(cat_bits, melting_bit, 4)
    cat_bits = _set_cat_bits(cat_bits, insect_bit, 6)
    return cat_bits


def get_melting_bit_ldr(Tw, ldr, v):
    """Finds melting layer from model temperature, ldr, and velocity.

    Args:
        Tw (ndarray): Wet bulb temperature, (n, m).
        ldr (ndarray): Linear depolarization ratio, (n, m).
        v (ndarray): Doppler velocity, (n, m).

    Returns:
        A (n, m) binary field indicating the melting layer (1=yes and 0=no).

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
    """Finds freezing region using the model temperature and melting layer.

    Sub-zero region is first derived from the model wet bulb temperature.
    It is then adjusted to start from the melting layer when we have such.
    Finally, a linear smoother is applied to combine the model and
    observations to avoid strong gradients in the zero-temperature line.

    Args:
        Tw (ndarray): Wet bulb temperature, (m, n).
        melting_bit (ndarray): Binary field indicating melting layer, (m, n).
        time (ndarray): Time vector, (m,).
        height (ndarray): Altitude vector, (n,).

    Returns:
        Binary field indicating the sub-zero region, (m, n).

    Notes:
        It is not clear how model temperature and melting layer should be
        ideally combined to determine the sub-zero region.

    """
    cold_bit = np.zeros(Tw.shape, dtype=int)
    ntime = time.shape[0]
    t0_alt = _get_t0_alt(Tw, height)
    mean_melting_height = np.zeros((ntime,))
    for ii in np.where(np.any(melting_bit, axis=1))[0]:
        mean_melting_height[ii] = np.median(
            height[np.where(melting_bit[ii, :])])
    m_final = np.copy(mean_melting_height)
    win = 240
    m_final[0] = mean_melting_height[0] or t0_alt[0]
    m_final[-1] = mean_melting_height[-1] or t0_alt[-1]
    for n in range(win, ntime-win):
        data_in_window = mean_melting_height[n-win:n+win+1]
        if not np.any(data_in_window):
            m_final[n] = t0_alt[n]
    ind = np.where(m_final > 0)[0]
    f = interp1d(time[ind], m_final[ind], kind='linear')
    tline = f(time)
    for ii, alt in enumerate(tline):
        cold_bit[ii, np.where(height > alt)[0]] = 1
    return cold_bit


def _set_cat_bits(cat_bits, bits_in, k):
    """ Updates categorize-bits array. """
    ind = np.where(bits_in)
    cat_bits[ind] = utils.bit_set(cat_bits[ind], k)
    return cat_bits


def _get_t0_alt(Tw, height):
    """ Interpolates altitudes where model temperature goes
        below freezing.

    Args:
        Tw (ndarray): Wet bulb temperature, (n, m).
        height (ndarray): Altitude vector, (m,).

    Returns:
        1D array containing the interpolated freezing altitudes.

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


def get_insect_bit(radar, Tw, *args):
    """ Returns insect probability and binary field indicating insects.

    Args:
        radar: A dict containing gridded radar fields
            ('Zh', 'ldr', 'width').
        Tw (ndarray): Wet bulb temperature.
        *args: Binary fields that are used to screen the
            insect probability. E.g. rain_bit, clutter_bit,
            melting_layer_bit, ...

    Returns:
        A 2-element tuple containing result of classification
        for each pixel (1=insect, 0=no) and insect probability
        (0-1).

    """
    insect_bit = np.zeros(Tw.shape, dtype=int)
    iprob = _insect_probability(radar['Zh'], radar['ldr'], radar['width'])
    iprob_screened = _screen_insects(iprob, Tw, *args)
    insect_bit[iprob_screened > 0.7] = 1  # limit should be optional argument
    return insect_bit, iprob_screened


def _insect_probability(z, ldr, width):
    """Finds insect probability from radar parameters.

    Args:
        z (ndarray): Radar echo.
        ldr (ndarray): Radar linear depolarization ratio.
        width (ndarray): Radar spectral width.

    Returns:
        Insect probability between 0-1 for all pixels.

    """
    def _insect_prob_ldr(z, ldr, z_loc=15, ldr_loc=-20):
        """Finds probability of insects, based on echo and ldr."""
        zp, ldrp = np.zeros(z.shape), np.zeros(z.shape)
        ind = ~z.mask
        zp[ind] = stats.norm.cdf(z[ind]*-1, loc=z_loc, scale=8)
        ind = ~ldr.mask
        ldrp[ind] = stats.norm.cdf(ldr[ind], loc=ldr_loc, scale=5)
        return zp * ldrp

    def _insect_prob_width(z, ldr, w, w_limit=0.06):
        """Finds (0, 1) probability of insects, based on spectral width."""
        i_prob = np.zeros(z.shape)
        temp_w = np.ones(z.shape)
        # pixels that have Z but no LDR
        ind = np.logical_and(ldr.mask, ~z.mask)
        temp_w[ind] = w[ind]
        i_prob[temp_w < w_limit] = 1
        return i_prob

    p1 = _insect_prob_ldr(z, ldr)
    p2 = _insect_prob_width(z, ldr, width)
    return p1 + p2


def _screen_insects(insect_prob, Tw, *args):
    """Screens insects by temperature and other misc. conditions.

    Args:
        insect_prob (ndarray): Insect probability, (m, n).
        Tw (ndarray): (m, n)
        *args (ndrray): Variable number of binary fields where 1
            means that insect probablity should be 0. Shape of these
            fields can be (m, n), or (m,) when whole profile
            will be flagged.

    """
    def _screen_insects_misc(insect_prob, *args):
        """Sets insect probability to 0, indicated by *args."""
        for arg in args:
            if arg.size == insect_prob.shape[0]:
                insect_prob[arg == 1, :] = 0
            else:
                insect_prob[arg == 1] = 0
        return insect_prob

    def _screen_insects_temp(insect_prob, Tw, t_lim=-5):
        """Removes insects from too cold temperatures."""
        insect_prob[Tw < (T0+t_lim)] = 0
        return insect_prob

    prob = np.copy(insect_prob)
    prob = _screen_insects_misc(prob, *args)
    prob = _screen_insects_temp(prob, Tw)
    return prob


def get_rain_bit(Z, time, time_buffer=5):
    """Find profiles affected by rain.

    Args:
        Z (ndarray): Radar echo with shape (m, n).
        time (ndarray): Time vector with shape (m,).
        time_buffer (float, optional): If profile includes rain,
            profiles measured **time_buffer** minutes before
            and after are also flagged to contain rain. Defaults to 5.

    Returns:
        A 1D binary array indicating profiles affected by
        rain (1=yes, 0=no).

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


def get_clutter_bit(v, rain_bit, ngates=10, vlim=0.05):
    """Estimates clutter from doppler velocity.

    Args:
        v (ndarray): Doppler velocity, (n, m).
        rain_bit (ndarray): A (n,) array indicating
            profiles affected by rain (1=yes, 0=no).
        vlim (float, optional): Velocity threshold.
            Default is 0.05 (m/s).

    Returns:
        2-D binary array containing pixels affected
        by clutter (1=yes, 0=no).

    """
    clutter_bit = np.zeros(v.shape, dtype=int)
    no_rain = np.where(rain_bit == 0)[0]
    ind = np.ma.where(np.abs(v[no_rain, 0:ngates]) < vlim)
    for n, m in zip(*ind):
        clutter_bit[no_rain[n], m] = 1
    return clutter_bit
