""" Classify gridded measurements. """

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
    melting_bit = get_melting_bit_ldr(model['Tw'], radar['ldr'], radar['v'],
                                      height, time)


    cat_bits = _set_cat_bits(cat_bits, melting_bit, 4)
    
    return cat_bits
    

def get_melting_bit_ldr(Tw, ldr, v, height, time):
    """ 
    Estimate melting layer from LDR data and model temperature 
    """

    melting_bit = np.zeros(Tw.shape, dtype=int)
    ldr_diff = np.diff(ldr, axis=1).filled(fill_value=0)
    v_diff = np.diff(v, axis=1).filled(fill_value=0)
    TRANGE = (-2, 5)  # find peak from this T range around T0

    for ii, tprof in enumerate(Tw):
        ind = np.where(np.logical_and(tprof > T0+TRANGE[0], tprof < T0+TRANGE[1]))[0]
        nind = len(ind)
        ldr_prof = ldr[ii, ind]
        ldr_dprof = ldr_diff[ii, ind]
        nldr = ma.count(ldr_prof)
        v_prof = v[ii, ind]
        v_dprof = v_diff[ii, ind]
        nv = ma.count(v_prof)
        alt = height[ind]
        ldr_p = np.argmax(ldr_prof)
        v_p = np.argmax(v_dprof)
        if nldr > 3 or nv > 3:
            try:
                ldr_top = droplet._get_top_ind(ldr_dprof, ldr_p, nind, 10, 2)
                ldr_base = droplet._get_base_ind(ldr_dprof, ldr_p, 10, 2)
                ldr_dd = ldr_prof[ldr_p] - ldr_prof[ldr_top]
                ldr_dd2 = ldr_prof[ldr_p] - ldr_prof[ldr_base]
                cond1 = ldr_dd > 4
                cond2 = ldr_prof[ldr_p] > -20
                cond3 = ldr_dd2 > 4
                cond4 = v_prof[ldr_base] < -2
                if cond1 and cond2 and cond3 and cond4:
                    melting_bit[ii, ind[ldr_p]:ind[ldr_top]+1] = 1
            except:
                try:
                    v_base = droplet.get_base_ind(v_dprof, v_p, 10, 2)
                    v_top = droplet.get_top_ind(v_dprof, v_p, nind, 10, 2)
                    v_dd = v_prof[v_top] - v_prof[v_base]
                    if (v_dd > 1) and (v_prof[v_base] < -2):
                        melting_bit[ii, ind[v_p-1:v_p+2]] = 2
                except:
                    continue
    return melting_bit


def _set_cat_bits(cat_bits, bits_in, k):
    """ Update categorize-bits array. """
    ind = np.where(bits_in)
    cat_bits[ind] = utils.bit_set(cat_bits[ind], k)
    return cat_bits
