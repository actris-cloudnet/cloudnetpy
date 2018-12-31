""" Classify gridded measurements. """

import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d
from scipy import stats
from cloudnetpy import droplet
from cloudnetpy import utils
from cloudnetpy.constants import T0


def fetch_cat_bits(radar, beta, Tw, time, height):
    """Classifies radar/lidar observations.

    Args:
        radar: A dict containing gridded radar fields
            ('Zh', 'v', 'ldr', 'width').
        beta (MaskedArray): Attenuated backscattering coefficient.
        Tw (ndarray): Wet bulb temperature.
        height (ndarray): 1D altitude vector.

    Returns: A dict containing the classification, 'cat_bits', where:
            - bit 0: Liquid droplets
            - bit 1: Falling hydrometeors
            - bit 2: Temperature < 0 Celsius
            - bit 3: Melting layer
            - bit 4: Aerosols
            - bit 5: Insects

        The dict contains also profiles containing rain
        and pixels contaminated by clutter.

    """
    bits = [None]*6
    is_rain = rain_from_radar(radar['Zh'], time)
    is_clutter = find_clutter(radar['v'], is_rain)    
    is_liquid, liquid_base, liquid_top = droplet.find_liquid(beta, height)
    bits[3] = find_melting_layer(Tw, radar['ldr'], radar['v'])
    bits[2] = find_freezing_region(Tw, bits[3], time, height)
    bits[0] = droplet.correct_liquid_top(radar['Zh'], Tw, bits[2],
                                         is_liquid, liquid_top, height)
    bits[5], insect_prob = find_insects(radar, Tw, bits[3], bits[0],
                                        is_rain, is_clutter)
    bits[1] = find_falling_hydrometeors(radar['Zh'], beta, is_clutter,
                                        bits[0], bits[5], Tw)
    bits[4] = find_aerosols(beta, bits[1], bits[0])
    cat_bits = _bits_to_integer(bits)
    return {'cat': cat_bits, 'rain': is_rain, 'liquid_base': liquid_base,
            'clutter': is_clutter, 'insect_prob': insect_prob}


def _bits_to_integer(bits):
    """Creates ndarray of integers from individual boolean fields.

    Args:
        bits (list): List of bit fields (of similar sizes!)
        to be saved in the resulting array of integers. bits[0]
        is saved as bit 1, bits[1] as bit 2, etc.

    Returns:
        Array of integers containing the information of the
        individual boolean fields.

    """
    int_array = np.zeros_like(bits[0], dtype=int)
    for n, bit in enumerate(bits):
        ind = np.where(bit)  # works also if bit is None
        int_array[ind] = utils.bit_set(int_array[ind].astype(int), n)
    return int_array


def find_melting_layer(Tw, ldr, v):
    """Finds melting layer from model temperature, ldr, and velocity.

    Args:
        Tw (ndarray): Wet bulb temperature, (n, m).
        ldr (ndarray): Linear depolarization ratio, (n, m).
        v (ndarray): Doppler velocity, (n, m).

    Returns:
        Boolean array denoting the melting layer.

    """
    def _slice(arg1, arg2, ii, ind):
        out1, out2 = arg1[ii, ind], arg2[ii, ind]
        return out1, out2, ma.count(out1)

    def _basetop(dprof, pind, nind, a=10, b=2):
        top = droplet.top_ind(dprof, pind, nind, a, b)
        base = droplet.base_ind(dprof, pind, a, b)
        return top, base

    melting_layer = np.zeros(Tw.shape, dtype=bool)
    ldr_diff = np.diff(ldr, axis=1).filled(0)
    v_diff = np.diff(v, axis=1).filled(0)
    trange = (-2, 5)  # find peak from this T range around T0
    for ii, tprof in enumerate(Tw):
        ind = np.where((tprof > T0+trange[0]) &
                       (tprof < T0+trange[1]))[0]
        nind = len(ind)
        ldr_prof, ldr_dprof, nldr = _slice(ldr, ldr_diff, ii, ind)
        v_prof, v_dprof, nv = _slice(v, v_diff, ii, ind)
        if nldr > 3 or nv > 3:
            ldr_p = np.argmax(ldr_prof)
            v_p = np.argmax(v_dprof)
            try:
                top, base = _basetop(ldr_dprof, ldr_p, nind)
                conds = (ldr_prof[ldr_p] - ldr_prof[top] > 4,
                         ldr_prof[ldr_p] - ldr_prof[base] > 4,
                         ldr_prof[ldr_p] > -20,
                         v_prof[base] < -2)
                if all(conds):
                    melting_layer[ii, ind[ldr_p]:ind[top]+1] = True
            except:  # just cach all exceptions
                try:
                    top, base = _basetop(v_dprof, v_p, nind)
                    diff = v_prof[top] - v_prof[base]
                    if diff > 1 and v_prof[base] < -2:
                        melting_layer[ii, ind[v_p-1:v_p+2]] = True
                except:  # failed whatever the reason
                    continue
    return melting_layer


def find_freezing_region(Tw, melting_layer, time, height):
    """Finds freezing region using the model temperature and melting layer.

    Sub-zero region is first derived from the model wet bulb temperature.
    It is then adjusted to start from the melting layer when we have such.
    Finally, a linear smoother is applied to combine the model and
    observations to avoid strong gradients in the zero-temperature line.

    Args:
        Tw (ndarray): Wet bulb temperature, (m, n).
        melting_layer (ndarray): Binary field indicating melting layer, (m, n).
        time (ndarray): Time vector, (m,).
        height (ndarray): Altitude vector, (n,).

    Returns:
        Boolean array denoting the sub-zero region.

    Notes:
        It is not clear how model temperature and melting layer should be
        ideally combined to determine the sub-zero region.

    """
    is_freezing = np.zeros(Tw.shape, dtype=bool)
    ntime = time.shape[0]
    t0_alt = _T0_alt(Tw, height)
    mean_melting_height = np.zeros((ntime,))
    for ii in np.where(np.any(melting_layer, axis=1))[0]:
        mean_melting_height[ii] = ma.median(height[melting_layer[ii, :]])
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
        is_freezing[ii, np.where(height > alt)[0]] = True
    return is_freezing


def _T0_alt(Tw, height):
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


def find_insects(radar, Tw, *args, prob_lim=0.8):
    """ Returns insect probability and binary field indicating insects.

    Args:
        radar: A dict containing gridded radar fields
            ('Zh', 'ldr', 'width').
        Tw (ndarray): Wet bulb temperature.
        *args: Binary fields that are used to screen the
            insect probability. E.g. rain, clutter,
            melting_layer, ...
        prob_lim (float, optional): Probability higher than
            this will lead to positive result. Default is 0.7.

    Returns:
        A 2-element tuple containing result of classification (bool)
        for each pixel and insect probability (0-1).

    """
    iprob = _insect_probability(radar['Zh'], radar['ldr'], radar['width'])
    iprob_screened = _screen_insects(iprob, Tw, *args)
    is_insects = iprob_screened > prob_lim
    return is_insects, iprob_screened


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
        temp_w = np.ones(z.shape)
        ind = ldr.mask & ~z.mask  # pixels that have Z but no LDR
        temp_w[ind] = w[ind]
        return (temp_w < w_limit).astype(int)

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
            fields can be (m, n), or (m,) when the whole profile
            is flagged.

    """
    def _screen_insects_misc(insect_prob, *args):
        """Sets insect probability to 0, indicated by *args."""
        for arg in args:
            if arg.size == insect_prob.shape[0]:
                insect_prob[arg, :] = 0
            else:
                insect_prob[arg] = 0
        return insect_prob

    def _screen_insects_temp(insect_prob, Tw, t_lim=-5):
        """Removes insects from too cold temperatures."""
        insect_prob[Tw < (T0+t_lim)] = 0
        return insect_prob

    prob = np.copy(insect_prob)
    prob = _screen_insects_misc(prob, *args)
    prob = _screen_insects_temp(prob, Tw)
    return prob


def rain_from_radar(Z, time, time_buffer=5):
    """Find profiles affected by rain.

    Args:
        Z (ndarray): Radar echo with shape (m, n).
        time (ndarray): Time vector with shape (m,).
        time_buffer (float, optional): If profile includes rain,
            profiles measured **time_buffer** minutes before
            and after are also flagged to contain rain. Defaults to 5.

    Returns:
        1-D boolean array denoting profiles affected by rain.

    """
    is_rain = ma.array(Z[:, 3] > 0, dtype=bool).filled(False)
    nprofs = len(time)
    step = utils.med_diff(time)*60
    nsteps = int(round(time_buffer/step/2))
    for ind in np.where(is_rain)[0]:
        i1 = max(0, ind-nsteps)
        i2 = min(ind+nsteps+1, nprofs)
        is_rain[i1:i2] = True
    return is_rain


def find_clutter(v, is_rain, ngates=10, vlim=0.05):
    """Estimates clutter from doppler velocity.

    Args:
        v (MaskedArray): Doppler velocity.
        is_rain (ndarray): 1-D boolean array indicating
            profiles affected by rain.
        vlim (float, optional): Velocity threshold.
            Smaller values are classified as clutter.
            Default is 0.05 (m/s).

    Returns:
        Boolean array denoting pixels contaminated by clutter.

    """
    is_clutter = np.zeros(v.shape, dtype=bool)
    tiny_velocity = (np.abs(v[:, :ngates]) < vlim).filled(False)
    is_clutter[:, :ngates] = (tiny_velocity.T*(~is_rain)).T
    return is_clutter


def find_falling_hydrometeors(Z, beta, is_clutter, is_liquid,
                              is_insects, Tw):
    """Finds falling hydrometeors.

    Args:
        Z (MaskedArray): Radar echo.
        beta (MaskedArray): Lidar echo.
        is_clutter (ndarray): Pixels contaminated by clutter.
        is_liquid (ndarray): Pixels containing droplets.
        is_insects (ndarray): Pixels containing insects.
        Tw (ndarray): Wet bulb temperature.

    Returns:
        Boolean array containing falling hydrometeros.

    """
    good_Z = ~Z.mask
    no_clutter = ~is_clutter
    no_insects = ~is_insects
    ice_from_lidar = ~beta.mask & ~is_liquid & (Tw < (T0-7))
    is_falling = (good_Z & no_clutter & no_insects) | ice_from_lidar
    return utils.filter_isolated_pixels(is_falling)


def find_aerosols(beta, is_falling, is_liquid):
    """Estimates aerosols from lidar backscattering.

    Aerosols are the unmasked pixels in the attenuated backscattering
    that are: (a) not falling, (b) not liquid droplets.

    Args:
        beta (MaskedArray): Attenuated backscattering coefficient.
        is_falling (ndarray): Binary array containing falling hydrometeors.
        is_liquid (ndarray): Binary array containing liquid droplets.

    Returns:
        Boolean array for aerosol classification.

    """
    return ~beta.mask & ~is_falling & ~is_liquid


def fetch_qual_bits(Z, beta, is_clutter, liq_atten):
    """Returns Cloudnet quality bits.

    Args:
        Z (MaskedArray): Radar echo.
        beta (MaskedArray): Attenuated backscattering.
        is_clutter (ndarray): Boolean array showing pixels
            contaminated by clutter.
        liq_atten (dict): Dictionary including boolean arrays
            'corr_bit' and 'ucorr_bit' that indicate where liquid
            attenuation was corrected and where it wasn't.

    Returns: Integer array containing the following bits:
            - bit 0: Pixel contains radar data.
            - bit 1: Pixel contains lidar data.
            - bit 2: Pixel contaminated by radar clutter.
            - bit 3: Molecular scattering present (currently not implemented!).
            - bit 4: Pixel was affected by liquid attenuation.
            - bit 5: Liquid attenuation was corrected.

    """
    bits = [None]*6
    bits[0] = ~Z.mask
    bits[1] = ~beta.mask
    bits[2] = is_clutter
    bits[4] = liq_atten['corr_bit'] | liq_atten['ucorr_bit']
    bits[5] = liq_atten['corr_bit']
    return _bits_to_integer(bits)
