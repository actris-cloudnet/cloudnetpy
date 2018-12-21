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
            - bit 1: Liquid droplets
            - bit 2: Falling hydrometeors
            - bit 3: Temperature < 0
            - bit 4: Melting layer
            - bit 5: Aerosols
            - bit 6: Insects

        The dict contains also profiles containing rain, 'rain_bit',
        and pixels contaminated by clutter, 'clutter_bit'.

    """
    bits = [None]*6
    bits[3] = get_melting_bit(Tw, radar['ldr'], radar['v'])
    bits[2] = get_cold_bit(Tw, bits[3], time, height)
    bits[0] = droplet.get_liquid_layers(beta, height)
    rain_bit = get_rain_bit(radar['Zh'], time)
    clutter_bit = get_clutter_bit(radar['v'], rain_bit)
    bits[5], insect_prob = get_insect_bit(radar, Tw, bits[3], bits[0],
                                          rain_bit, clutter_bit)
    bits[1] = get_falling_bit(radar['Zh'], clutter_bit, bits[5])
    bits[4] = get_aerosol_bit(beta, bits[1], bits[0])
    cat_bits = _bits_to_integer(bits)
    return {'cat_bits': cat_bits, 'rain_bit': rain_bit,
            'clutter_bit': clutter_bit, 'insect_prob': insect_prob}


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
    for n, bit in enumerate(bits, 1):
        ind = np.where(bit)
        int_array[ind] = utils.bit_set(int_array[ind].astype(int), n)
    return int_array


def get_melting_bit(Tw, ldr, v):
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
        top = droplet.get_top_ind(dprof, pind, nind, a, b)
        base = droplet.get_base_ind(dprof, pind, a, b)
        return top, base

    melting_bit = np.zeros(Tw.shape, dtype=bool)
    ldr_diff = np.diff(ldr, axis=1).filled(0)
    v_diff = np.diff(v, axis=1).filled(0)
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
                conds = (ldr_prof[ldr_p] - ldr_prof[top] > 4,
                         ldr_prof[ldr_p] - ldr_prof[base] > 4,
                         ldr_prof[ldr_p] > -20,
                         v_prof[base] < -2)
                if all(conds):
                    melting_bit[ii, ind[ldr_p]:ind[top]+1] = True
            except:  # just cach all exceptions
                try:
                    top, base = _basetop(v_dprof, v_p, nind)
                    diff = v_prof[top] - v_prof[base]
                    if diff > 1 and v_prof[base] < -2:
                        melting_bit[ii, ind[v_p-1:v_p+2]] = True
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
        Boolean array denoting the sub-zero region.

    Notes:
        It is not clear how model temperature and melting layer should be
        ideally combined to determine the sub-zero region.

    """
    cold_bit = np.zeros(Tw.shape, dtype=bool)
    ntime = time.shape[0]
    t0_alt = _get_T0_alt(Tw, height)
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
        cold_bit[ii, np.where(height > alt)[0]] = True
    return cold_bit


def _get_T0_alt(Tw, height):
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


def get_insect_bit(radar, Tw, *args, prob_lim=0.7):
    """ Returns insect probability and binary field indicating insects.

    Args:
        radar: A dict containing gridded radar fields
            ('Zh', 'ldr', 'width').
        Tw (ndarray): Wet bulb temperature.
        *args: Binary fields that are used to screen the
            insect probability. E.g. rain_bit, clutter_bit,
            melting_layer_bit, ...
        prob_lim (float, optional): Probability higher than
            this will lead to positive result. Default is 0.7.

    Returns:
        A 2-element tuple containing result of classification (bool)
        for each pixel and insect probability (0-1).

    """
    insect_bit = np.zeros(Tw.shape, dtype=bool)
    iprob = _insect_probability(radar['Zh'], radar['ldr'], radar['width'])
    iprob_screened = _screen_insects(iprob, Tw, *args)
    insect_bit[iprob_screened > prob_lim] = True
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
            fields can be (m, n), or (m,) when the whole profile
            is flagged.

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
        1-D boolean array denoting profiles affected by rain.

    """
    nprofs = len(time)
    rain_bit = np.zeros(nprofs, dtype=bool)
    rain_bit[Z[:, 3] > 0] = 1
    step = utils.med_diff(time)*60*60  # minutes
    nsteps = int(round(time_buffer*60/step/2))
    for ind in np.where(rain_bit)[0]:
        i1 = max(0, ind-nsteps)
        i2 = min(ind+nsteps+1, nprofs)
        rain_bit[i1:i2] = True
    return rain_bit


def get_clutter_bit(v, rain_bit, ngates=10, vlim=0.05):
    """Estimates clutter from doppler velocity.

    Args:
        v (MaskedArray): Doppler velocity.
        rain_bit (ndarray): 1-D boolean array indicating
            profiles affected by rain.
        vlim (float, optional): Velocity threshold.
            Smaller values are classified as clutter.
            Default is 0.05 (m/s).

    Returns:
        Boolean array denoting pixels contaminated by clutter.

    """
    clutter_bit = np.zeros(v.shape, dtype=bool)
    tiny_velocity = (np.abs(v[:, :ngates]) < vlim).filled(False)
    clutter_bit[:, :ngates] = (tiny_velocity.T*(~rain_bit)).T
    return clutter_bit


def get_falling_bit(Z, clutter_bit, insect_bit):
    """Finds falling hydrometeors.

    Args:
        Z (MaskedArray): Radar echo.
        clutter_bit (ndarray): Binary field of clutter.
        insect_bit (ndarray): Binary field of insects.

    Returns:
        Boolean array containing falling hydrometeros.

    """
    falling_bit = ~Z.mask & ~clutter_bit & ~insect_bit
    falling_bit = utils.filter_isolated_pixels(falling_bit)
    return falling_bit


def get_aerosol_bit(beta, falling_bit, droplet_bit):
    """Estimates aerosols from lidar backscattering.

    Aerosols are the unmasked pixels in the attenuated backscattering
    that are: (a) not falling, (b) not liquid droplets.

    Args:
        beta (MaskedArray): Attenuated backscattering coefficient.
        falling_bit (ndarray): Binary array containing falling hydrometeors.
        droplet_bit (ndarray): Binary array containing liquid droplets.

    Returns:
        Boolean array for aerosol classification.

    """
    return ~beta.mask & ~falling_bit & ~droplet_bit


def fetch_qual_bits(Z, beta, clutter_bit, atten):
    """Returns Cloudnet quality bits.

    Args:
        Z (MaskedArray): Radar echo.
        beta (MaskedArray): Attenuated backscattering.
        clutter_bit (ndarray): Boolean array showing pixels
            contaminated by clutter.
        atten (dict): Dictionary including boolean arrays
            'liq_atten_corr_bit' and 'liq_atten_ucorr_bit'
            that indicate where liquid attenuation was corrected
            and where it wasn't.

    Returns: Integer array containing the following bits:
            - bit 1: Pixel contains radar data.
            - bit 2: Pixel contains lidar data.
            - bit 3: Pixel contaminated by radar clutter.
            - bit 4: Molecular scattering present (currently not implemented!).
            - bit 5: Pixel was affected by liquid attenuation.
            - bit 6: Liquid attenuation was corrected.

    """
    bits = [None]*6
    bits[0] = (~Z.mask).astype(int)
    bits[1] = (~beta.mask).astype(int)
    bits[2] = clutter_bit
    bits[4] = atten['liq_atten_corr_bit'] | atten['liq_atten_ucorr_bit']
    bits[5] = atten['liq_atten_corr_bit']
    return _bits_to_integer(bits)
