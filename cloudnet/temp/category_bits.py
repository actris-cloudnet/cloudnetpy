
import numpy as np
import numpy.ma as ma


def fetch_cat_bits(model_time, model_height, model, radar, lidar, time, height, vfold, Tw):
    """ Experimental classification based on lidar and LDR-supported radar data """
    
    cat_bits = np.zeros(Tw.shape, dtype=int)
    melting_bit = get_melting_bit_ldr(Tw, radar['ldr'], radar['v'], height, time)
    cold_bit = get_cold_bit(Tw, melting_bit, time, height)
    cloud_bit, cloud_base, cloud_top = droplet.get_liquid_layers(lidar['beta'], height)
    droplet_bit = droplet.correct_liquid_cloud_top(radar['Zh'], lidar['beta'], Tw, cold_bit, cloud_bit, cloud_top, height)
    rain_bit = droplet.get_rain_bit(radar['Zh'], time)
    v_sigma = droplet.estimate_v_sigma(radar['v'])
    clutter_bit = droplet.get_radar_clutter(rain_bit, radar['v'], v_sigma)
    insect_bit, insect_prob = get_insect_bit(height, radar['ldr'], radar['Zh'], radar['v'], radar['width'],
                                             clutter_bit, cold_bit, rain_bit, melting_bit, droplet_bit, Tw)
    falling_bit = get_falling_bit(radar['Zh'], clutter_bit, insect_bit)
    aerosol_bit = get_aerosol_bit(lidar['beta'], falling_bit, droplet_bit, cold_bit)
    cat_bits = set_cat_bits(cat_bits, droplet_bit, 1)
    cat_bits = set_cat_bits(cat_bits, falling_bit, 2)
    cat_bits = set_cat_bits(cat_bits, cold_bit, 3)
    cat_bits = set_cat_bits(cat_bits, melting_bit, 4)
    cat_bits = set_cat_bits(cat_bits, aerosol_bit, 5)
    cat_bits = set_cat_bits(cat_bits, insect_bit, 6)
    return cat_bits, insect_prob, droplet_bit, rain_bit, melting_bit, clutter_bit


def get_aerosol_bit(beta, falling_bit, droplet_bit, cold_bit):
    """ Estimate aerosols from lidar measured beta """
    aerosol_bit = np.zeros(cold_bit.shape, dtype=int)
    mazk = (falling_bit==0) & (droplet_bit==0) & (~beta.mask) 
    aerosol_bit[mazk] = 1
    return aerosol_bit


def insect_prob_ldr(z, ldr):
    """ Probability that pixel is insect, based on Z and LDR values """
    zp = np.zeros(z.shape)
    ldrp = np.zeros(z.shape)
    ind = ~z.mask
    zp[ind] = stats.norm.cdf(z[ind]*-1, loc=15, scale=8)
    ind = ~ldr.mask
    ldrp[ind] = stats.norm.cdf(ldr[ind], loc=-20, scale=5)
    return zp * ldrp


def insect_prob_width(z, ldr, v, w):
    """ (0, 1) Probability that pixel is insect, based on WIDTH values """
    W_LIM = 0.07
    i_prob = np.zeros(z.shape) 
    temp_w = np.ones(z.shape)     
    # pixels that have Z but no LDR:
    ind = np.logical_and(ldr.mask, ~z.mask)
    temp_w[ind] = w[ind]
    i_prob[temp_w <= W_LIM] = 1
    return i_prob


def get_insect_bit(height, ldr, Z, v, width, clutter_bit, cold_bit, rain_bit, melting_bit, droplet_bit, Tw):
    """ Estimation of insect probability from radar Z, LDR, and WIDTH """

    insect_bit = np.zeros_like(cold_bit)    
    
    # probabiliy of insects from LDR-pixels
    p1 = insect_prob_ldr(Z, ldr)

    # probability of insects from pixels without LDR 
    p2 = insect_prob_width(Z, ldr, v, width)
    
    p_ins = p1 + p2
    
    p_ins[rain_bit==1,:] = 0
    #p_ins[cold_bit==1] = 0
    p_ins[Tw < (T0)-3] = 0
    p_ins[droplet_bit==1] = 0
    p_ins[melting_bit==1] = 0
    
    insect_bit[p_ins>0.8] = 1

    return insect_bit, p_ins


def get_cold_bit(Tw, melting_bit, time, height):
    """ Adjust cold bit so that it starts from
    the melting layer when we have it
    """

    cold_bit = np.zeros(Tw.shape, dtype=int)

    ntime = time.shape[0]
    
    # get model T0-line
    T0_alt = droplet.get_T0_alt(Tw, height, T0)

    # mean melting height
    mean_melting_height = np.zeros((ntime,))
    for ii in np.where(np.any(melting_bit, axis=1))[0]:
        mean_melting_height[ii] = np.median(height[np.where(melting_bit[ii,:])])

    m_final = np.copy(mean_melting_height)
    win = 240

    # ensure we have data points on the edges
    m_final[0] = mean_melting_height[0] or T0_alt[0][0]
    m_final[-1] = mean_melting_height[-1] or T0_alt[0][-1]
    
    for n in range (win, ntime-win):
        data_in_window = mean_melting_height[n-win:n+win+1]
        # no data in whole time window -> use model data
        if not np.any(data_in_window):
            m_final[n] = T0_alt[n]

    ind = np.where(m_final > 0)[0]
    f = interp1d(time[ind], m_final[ind], kind='linear')
    mi = f(time)

    # set cold bit according to melting layer
    for ii, alt in enumerate(mi):
        ind = np.where(height>alt)[0][0]
        cold_bit[ii,ind:] = 1
    
    return cold_bit


def set_cat_bits(cat_bits, bits_in, k):
    ind = np.where(bits_in)
    cat_bits[ind] = utils.set_bit(cat_bits[ind], k) 
    return cat_bits


def get_melting_bit_ldr(Tw, ldr, v, height, time):
    """ 
    Estimate melting layer from LDR data and model temperature 
    """
    melting_bit = np.zeros(Tw.shape, dtype=int)
    ldr_diff = np.diff(ldr,axis=1).filled(fill_value=0)
    TRANGE = (-2, 5)  # find peak from this T range around T0
    THRES = 0.2
    for ii, tprof in enumerate(Tw):
        ind = np.where(np.logical_and(tprof > T0+TRANGE[0], tprof < T0+TRANGE[1]))[0]
        nind = ind.shape[0]
        ldr_prof = ldr[ii, ind]
        ldr_dprof = ldr_diff[ii, ind]
        nldr = ma.count(ldr_prof)
        v_prof = v[ii, ind]
        nv = ma.count(v_prof)
        alt = height[ind]
        p = np.argmax(ldr_prof)        
        if nldr > 5 and nv > 5:
            try:
                n_above_top = min(nind-p-1, 6)
                top = droplet.get_top_ind(ldr_dprof, p, nind, n_above_top, 2)
            except:
                continue
            v_dd = v_prof[top] - v_prof[p]         # should be positive, ~1-3 m/s change
            ldr_dd = ldr_prof[p] - ldr_prof[top]   # should be also positive, around ~10-20 change
            nempty = ma.count_masked(ldr_prof[p:top])            
            if v_dd > 0.8 and ldr_dd > 7 and nempty == 0:
                melting_bit[ii, ind[p]:ind[top]+1] = 1                
    return melting_bit


def get_falling_bit(Z, clutter_bit, insect_bit):
    falling_bit = np.zeros_like(clutter_bit)
    falling_bit[~Z.mask] = 1 
    falling_bit[clutter_bit == 1] = 0
    falling_bit[insect_bit == 1] = 0
    falling_bit = utils.filter_isolated_pixels(falling_bit)
    return falling_bit
