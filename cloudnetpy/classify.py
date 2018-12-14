""" Classify gridded measurements. """

import sys
import numpy as np
import numpy.ma as ma
import droplet
import utils
from atmos import T0


def fetch_cat_bits(radar, lidar, model, time, height, vfold):
    """ Experimental classification based on lidar and LDR-supported radar data """

    cat_bits = np.zeros(model['Tw'].shape, dtype=int)

    if 'ldr' and 'v' not in radar:
        raise KeyError('Needs LDR and doppler velocity.')
    melting_bit = get_melting_bit_ldr(model['Tw'], radar['ldr'], radar['v'])

    cat_bits = _set_cat_bits(cat_bits, melting_bit, 4)

    return cat_bits


def get_melting_bit_ldr(Tw, ldr, v):
    """
    Estimate melting layer from LDR data and model temperature
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
    TRANGE = (-2, 5)  # find peak from this T range around T0
    for ii, tprof in enumerate(Tw):
        ind = np.where(np.logical_and(tprof > T0+TRANGE[0],
                                      tprof < T0+TRANGE[1]))[0]
        nind = len(ind)
        ldr_prof, ldr_dprof, nldr = _slice(ldr, ldr_diff, ii, ind)
        v_prof, v_dprof, nv = _slice(v, v_diff, ii, ind)        
        ldr_p = np.argmax(ldr_prof)
        v_p = np.argmax(v_dprof)
        if nldr > 3 or nv > 3:
            try:
                ldr_top, ldr_base = _basetop(ldr_dprof, ldr_p, nind)                
                ldr_dd = ldr_prof[ldr_p] - ldr_prof[ldr_top]
                ldr_dd2 = ldr_prof[ldr_p] - ldr_prof[ldr_base]
                conds = (ldr_dd > 4,
                         ldr_prof[ldr_p] > -20,
                         ldr_dd2 > 4,
                         v_prof[ldr_base] < -2)
                if all(conds):
                    melting_bit[ii, ind[ldr_p]:ind[ldr_top]+1] = 1
            except:
                try:
                    v_top, v_base = _basetop(v_dprof, v_p, nind)
                    v_dd = v_prof[v_top] - v_prof[v_base]
                    if v_dd > 1 and v_prof[v_base] < -2:
                        melting_bit[ii, ind[v_p-1:v_p+2]] = 2
                except:
                    continue
    return melting_bit


def _set_cat_bits(cat_bits, bits_in, k):
    """ Update categorize-bits array. """
    ind = np.where(bits_in)
    cat_bits[ind] = utils.bit_set(cat_bits[ind], k)
    return cat_bits
