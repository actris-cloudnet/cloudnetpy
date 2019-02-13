import sys
import netCDF4
import numpy as np
import numpy.ma as ma
from scipy.interpolate import interp1d, RectBivariateSpline
from scipy import stats, ndimage
import time
import calendar
import uuid
import math
from datetime import datetime, timezone
#import matplotlib.pyplot as plt
import ncf
from ncf import CnetVar, err_name, bias_name
import bits
import droplet
import lwc

T0 = 273.16 # triple point of water (0.01C)

def generate_categorize(radar_file, lidar_file, model_file, mwr_file, output_file, aux):

    # define constants
    VERSION = 11.0 # version of this software
    BETA_BIAS = 3.0
    BETA_ERROR = 0.5
    TIME_RESOLUTION = 30 # fixed time resolution in seconds
    Z_BIAS = 1.0
    LWP_FRAC_ERROR = 0.25
    LWP_LIN_ERROR = 20
    GAS_ATTEN_PREC = 0.1

    # read input files
    rad, rad_vrs = load_nc(radar_file)
    lid, lid_vrs = load_nc(lidar_file)
    mod, mod_vrs = load_nc(model_file)
    mwr, mwr_vrs = load_nc(mwr_file)
    
    # get radar frequency. At the moment it is either 35 or 94 GHz
    # This need to be generalized later for other frequencies
    freq, is35 = ncf.get_radar_freq(rad_vrs, 'frequency')

    # create universal grid (time, height)
    height, alt_site = get_height(rad_vrs['altitude'][:], lid_vrs['altitude'][:], rad_vrs['range'][:])
    time = get_time(TIME_RESOLUTION)
    dheight = ncf.med_diff(height)

    # date and site location
    dvec = ncf.get_date(rad)
    lat, lon = get_site_location(rad)
    
    # average radar variables in time
    fields = ['Zh', 'v', 'ldr', 'width']
    radar = fetch_radar(rad_vrs, fields, time)
    vfold = rad_vrs['NyquistVelocity'][:]
    
    # average lidar variables in time and height
    fields = ['beta']
    lidar = fetch_lidar(lid_vrs, fields, time, height)
    
    # read and interpolate mwr variables
    time_lwp, lwp, lwp_error = get_lwp(mwr_vrs, LWP_FRAC_ERROR, LWP_LIN_ERROR)
    lwp_i = interpolate_lwp(time_lwp, lwp, time)
    lwp_error_i = interpolate_lwp(time_lwp, lwp_error, time)

    # read and interpolate model variables
    fields = ['temperature', 'pressure', 'q', 'uwind', 'vwind', 'rh', 'gas_atten', 'specific_gas_atten', 'specific_saturated_gas_atten', 'specific_liquid_atten']
    model, model_time, model_height = fetch_model(mod_vrs, fields, alt_site, is35)
    fields_i = ['temperature', 'pressure', 'rh', 'gas_atten', 'specific_gas_atten', 'specific_saturated_gas_atten', 'specific_liquid_atten']
    model_i = interpolate_model(model, fields_i, model_time, model_height, time, height)

    # calculate wet bulb temperature
    Tw = wet_bulb(model_i['temperature'], model_i['pressure'], model_i['rh'])

    # classify measurements
    cat_bits, p_ins, droplet_bit, rain_bit, melting_bit, clutter_bit = fetch_cat_bits(model_time, model_height, model, radar,
                                                                                      lidar, time, height, vfold, Tw)
    
    # gas attenuation
    gas_atten = get_gas_attenuation(dheight, droplet_bit, model_i['gas_atten'], model_i['specific_gas_atten'],
                                    model_i['specific_saturated_gas_atten'])

    # liquid attenuation
    liq_atten, liq_atten_err, corr_atten_bit, uncorr_atten_bit = get_liquid_attenuation(lwp_i, lwp_error_i, droplet_bit, model_i['temperature'],
                                                                                        model_i['pressure'], model_i['specific_liquid_atten'],
                                                                                        dheight, rain_bit, melting_bit)
    # quality bits
    qual_bits = fetch_qual_bits(radar['Zh'], lidar['beta'], clutter_bit, corr_atten_bit, uncorr_atten_bit)
    
    # correct Z for attenuation
    Z_corrected = correct_attenuation(radar['Zh'], gas_atten, liq_atten)

    # calculate Z sensitivity and error
    Z_sens, Z_err = fetch_Z_errors(radar, rad_vrs['range'][:], time, freq, clutter_bit, gas_atten, GAS_ATTEN_PREC, liq_atten_err)
    
    # Collect variables for output
    cat_vars = {'height':height, 'time':time, 'latitude':lat, 'longitude':lon, 'altitude':alt_site, 
                'radar_frequency':freq, 'lidar_wavelength':lid_vrs['wavelength'][:], 
                'beta':lidar['beta'], 'beta_error':BETA_ERROR, 'beta_bias':BETA_BIAS, 
                'Z':Z_corrected, 'v':radar['v'], 'width':radar['width'], 'ldr':radar['ldr'], 
                'Z_bias':Z_BIAS, 'temperature':model['temperature'], 'pressure':model['pressure'], 
                'specific_humidity':model['q'], 'uwind':model['uwind'], 'vwind':model['vwind'], 
                'model_height':model_height, 'model_time':model_time, 'category_bits':cat_bits,
                'Tw':Tw, 'insect_probability':p_ins, 'radar_gas_atten': gas_atten,
                'radar_liquid_atten': liq_atten, 'lwp':lwp_i, 'lwp_error':lwp_error_i,
                'quality_bits':qual_bits, 'Z_error':Z_err, 'Z_sensitivity':Z_sens} 

    all_obs = cat_Cnet_vars(cat_vars, dvec, TIME_RESOLUTION)

    # create categorize file and save all data
    save_cat(output_file, rad, time, height, model_time, model_height, all_obs, dvec, VERSION, aux)


def fetch_Z_errors(radar, radar_range, time, freq, clutter_bit, gas_atten, GAS_ATTEN_PREC, liq_atten_err):

    Z = radar['Zh']
    
    # sensitivity
    Z_power = Z - 20*np.log10(radar_range)
    Z_power_list = np.sort(Z_power.compressed())
    Z_power_min = Z_power_list[int(np.floor(len(Z_power_list)/1000))]
    Z_sensitivity = Z_power_min + 20*np.log10(radar_range)
    Z_sensitivity = Z_sensitivity + np.mean(gas_atten,axis=0)

    Zc = ma.masked_where(clutter_bit==0, Z, copy=True)
    Zc = ma.median(Zc,axis=0)
    ind = ~Zc.mask
    Z_sensitivity[ind] = Zc[ind]
    
    # precision
    dwell_time = ncf.med_diff(time)*3600
    independent_pulses = (dwell_time*4*np.sqrt(math.pi)*freq*1e9/3e8) * radar['width']
    Z_precision = 4.343*(1.0/np.sqrt(independent_pulses) + 10**(0.1*(Z_power_min-Z_power))/3)

    # Z error
    Z_error = np.sqrt( (gas_atten*GAS_ATTEN_PREC)**2 + (liq_atten_err)**2 + (Z_precision)**2  )
    
    return Z_sensitivity, Z_error


    
def fetch_qual_bits(Z, beta, clutter_bit, corr_atten_bit, uncorr_atten_bit):
    """ Quality bits """
    qual_bits = np.zeros_like(clutter_bit)
    lidar_bit = np.zeros_like(clutter_bit)
    radar_bit = np.zeros_like(clutter_bit)

    radar_bit[~Z.mask] = 1
    lidar_bit[~beta.mask] = 1
    
    set_cat_bits(qual_bits, radar_bit, 1)
    set_cat_bits(qual_bits, lidar_bit, 2)
    set_cat_bits(qual_bits, clutter_bit, 3)
    # 4 = molecular scattering, still missing
    set_cat_bits(qual_bits, corr_atten_bit, 5)
    set_cat_bits(qual_bits, uncorr_atten_bit, 5)
    set_cat_bits(qual_bits, corr_atten_bit, 6)
    
    return qual_bits


def correct_attenuation(Z, gas_atten, liquid_atten):
    """ Correct radar backscattering for attenuation """
    Z_corrected = ma.copy(Z) + gas_atten
    ind = ~liquid_atten.mask
    Z_corrected[ind] = Z_corrected[ind] + liquid_atten[ind]
    return Z_corrected


def get_liquid_attenuation(lwp, lwp_error, droplet_bit, temperature, pressure, specific_liquid_atten, dheight, rain_bit, melting_bit):
    """ approximation of a liquid attenuation in a profile """
    # init 
    msize = temperature.shape
    lwc_adiabatic = ma.zeros(msize)
    lwc_error = ma.zeros(msize)
    lwp_boxes = ma.zeros(msize)
    lwp_boxes_error = ma.zeros(msize)
    liquid_attenuation = ma.zeros(msize)
    liquid_attenuation_error = ma.zeros(msize)
    corr_atten_bit = np.zeros_like(droplet_bit)
    uncorr_atten_bit = np.zeros_like(droplet_bit)

    # important profiles
    is_liquid = np.any(droplet_bit, axis=1)
    is_lwp = np.isfinite(lwp)
    
    for ii in np.where(np.logical_and(is_lwp, is_liquid))[0]:
        (bases, tops) = ncf.tops_and_bases(droplet_bit[ii,:])
        for base, top in zip(bases, tops):
            npoints = top - base + 1
            idx = np.arange(npoints) + base
            dlwc_dz = lwc.theory_adiabatic_lwc(temperature[ii,base], pressure[ii,base])
            lwc_adiabatic[ii,idx] = dlwc_dz * dheight * 1000 * (np.arange(npoints)+1)
            lwc_error[ii,idx] = dlwc_dz # unnormalised

        lwp_boxes[ii,:] = lwp[ii] * lwc_adiabatic[ii,:] / np.sum(lwc_adiabatic[ii,:])
        lwp_boxes_error[ii,:] = lwp_error[ii] * lwc_error[ii,:] / np.sum(lwc_error[ii,:])

    for ii in np.where(~is_lwp)[0]:
        lwp_boxes[ii,droplet_bit[ii,:]==1] = None

    liquid_attenuation[:,1:] = 0.002 * ma.cumsum(lwp_boxes[:,0:-1] * specific_liquid_atten[:,0:-1], axis=1)
    liquid_attenuation_error[:,1:] = 0.002 * np.cumsum(lwp_boxes_error[:,0:-1] * specific_liquid_atten[:,0:-1], axis=1)

    # remove rainy profiles
    liquid_attenuation[rain_bit==1,:] = None

    # remove melting profiles
    above_melting = np.cumsum(melting_bit>0,axis=1)
    liquid_attenuation[above_melting>=1] = None    
    
    # mask invalid and zero values
    liquid_attenuation = ma.masked_invalid(liquid_attenuation)
    liquid_attenuation = ma.masked_equal(liquid_attenuation,0)

    # bit indicating attenuation that was corrected
    corr_atten_bit[~liquid_attenuation.mask] = 1

    # bit indicating attenuation that was NOT corrected
    uncorr_atten_bit[rain_bit==1,:] = 1
    uncorr_atten_bit[above_melting>=1] = 1

    return liquid_attenuation, liquid_attenuation_error, corr_atten_bit, uncorr_atten_bit

        
def get_gas_attenuation(dheight, droplet_bit, gas_atten, specific_gas_atten, specific_saturated_gas_atten):
    """ Calculate gas attenuation. Inside liquid droplets assume saturation."""
    gas_attenuation = np.zeros(droplet_bit.shape, dtype=float)
    true_specific_gas_atten = np.copy(specific_gas_atten)
    true_specific_gas_atten[droplet_bit==1] = specific_saturated_gas_atten[droplet_bit==1]
    gas_attenuation = ncf.expand_to_alt(gas_atten[:,0], droplet_bit.shape[1])
    gas_attenuation[:,1:] = gas_attenuation[:,1:] + 2.0*np.cumsum(true_specific_gas_atten[:,0:-1],axis=1)*dheight*0.001
    return gas_attenuation


def interpolate_model(model, fields, model_time, model_height, time, height):
    """ Interpolate model fields into universal time/height grid """
    out = {}
    for field in fields:
        out[field] = interpolate_2d(model_time, model_height, model[field], time, height)
    return out
    

def fetch_cat_bits(model_time, model_height, model, radar, lidar, time, height, vfold, Tw):
    """ Experimental classification based on lidar and LDR-supported radar data """
    
    cat_bits = np.zeros(Tw.shape, dtype=int)

    # melting bit
    # -----------
    if 'ldr' and 'v' in radar:
        melting_bit = get_melting_bit_ldr(Tw, radar['ldr'], radar['v'], height, time)
    else:
        print('Need another method if ldr is not available..')

    # cold bit
    # --------
    cold_bit = get_cold_bit(Tw, melting_bit, time, height)
    
    # small droplet bit
    # -----------------
    cloud_bit, cloud_base, cloud_top = droplet.get_liquid_layers(lidar['beta'], height)
    droplet_bit = droplet.correct_liquid_cloud_top(radar['Zh'], lidar['beta'], Tw, cold_bit, cloud_bit, cloud_top, height)
    
    # falling bit and insect bit
    # --------------------------
    rain_bit = droplet.get_rain_bit(radar['Zh'], time)    
    v_sigma = droplet.estimate_v_sigma(radar['v'])
    clutter_bit = droplet.get_radar_clutter(rain_bit, radar['v'], v_sigma)
    if 'ldr' in radar:
        insect_bit, insect_prob = get_insect_bit(height, radar['ldr'], radar['Zh'], radar['v'], radar['width'],
                                                   clutter_bit, cold_bit, rain_bit, melting_bit, droplet_bit)
        falling_bit = get_falling_bit(radar['Zh'], clutter_bit, insect_bit)
    else:
        print('Need another method if ldr is not available..')

    # aerosol bit
    # -----------
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


def get_insect_bit(height, ldr, Z, v, width, clutter_bit, cold_bit, rain_bit, melting_bit, droplet_bit):
    """ Estimation of insect probability from radar Z, LDR, and WIDTH """

    insect_bit = np.zeros_like(cold_bit)
    
    # probabiliy of insects from LDR-pixels
    p1 = insect_prob_ldr(Z, ldr)

    # probability of insects from pixels without LDR 
    p2 = insect_prob_width(Z, ldr, v, width)
    
    p_ins = p1 + p2
    
    p_ins[rain_bit==1,:] = 0
    #p_ins[cold_bit==1] = 0
    p_ins[droplet_bit==1] = 0
    p_ins[melting_bit==1] = 0
    
    insect_bit[p_ins>0.5] = 1
    
    return insect_bit, p_ins


def filter_isolated_bits(array):
    """ Return array with completely isolated single cells removed """
    filtered_array = ma.copy(array)
    id_regions, num_ids = ndimage.label(filtered_array, structure=np.ones((3,3)))
    id_sizes = np.array(ndimage.sum(array, id_regions, range(num_ids + 1)))
    area_mask = (id_sizes == 1)
    filtered_array[area_mask[id_regions]] = 0
    return filtered_array


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
    cat_bits[ind] = bits.set_bit(cat_bits[ind], k) 
    return cat_bits


def get_melting_bit_ldr(Tw, ldr, v, height, time):
    """ 
    Estimate melting layer from LDR data and model temperature 
    """

    melting_bit = np.zeros(Tw.shape, dtype=int)

    ldr_diff = np.diff(ldr,axis=1).filled(fill_value=0)
    TRANGE = (-2, 5) # find peak from this T range around T0
    THRES = 0.2

    for ii, tprof in enumerate(Tw):

        ind = np.where(np.logical_and(tprof > T0+TRANGE[0], tprof < T0+TRANGE[1]))[0]
        nind = ind.shape[0]

        # ldr
        ldr_prof = ldr[ii, ind]
        ldr_dprof = ldr_diff[ii, ind]
        nldr = ma.count(ldr_prof)

        # v
        v_prof = v[ii, ind]
        nv = ma.count(v_prof)

        # altitude 
        alt = height[ind]

        # peak value in ldr
        p = np.argmax(ldr_prof)
        
        if (nldr > 5) and (nv > 5):

            try:
                n_above_top = min(nind-p-1, 6)
                top = droplet.get_top_ind(ldr_dprof, p, nind, n_above_top, 2)
            except:
                continue

            v_dd = v_prof[top] - v_prof[p]         # should be positive, ~1-3 m/s change
            ldr_dd = ldr_prof[p] - ldr_prof[top]   # should be also positive, around ~10-20 change

            nempty = ma.count_masked(ldr_prof[p:top])
            
            if ( (v_dd>0.8) and (ldr_dd>7) and (nempty==0) ):
                melting_bit[ii, ind[p]:ind[top]+1] = 1
                
    return melting_bit


def get_falling_bit(Z, clutter_bit, insect_bit):
    falling_bit = np.zeros_like(clutter_bit)
    falling_bit[~Z.mask] = 1 
    falling_bit[clutter_bit==1] = 0
    falling_bit[insect_bit==1] = 0
    falling_bit = filter_isolated_bits(falling_bit)
    return falling_bit


def wet_bulb(Tdry, p, rh):
    """ This function calculates the wet bulb temperature given temperature,
    pressure and relative humidity
    Algorithm based on a Taylor expansion of a simple expression for
    the saturated vapour pressure:
    wet_bulb_experiment(Tdry, p, rh)
    Tdry: K
    p: Pa
    rh: 0-1
    """

    def e_saturation_liquid(T):
        return 6.11e2 * np.exp(17.269 * (T-273.16) / (T-35.86) )
    

    def dew_point(e_saturation):
        factor = np.log(e_saturation/6.11e2)/17.269
        return (35.86*factor-273.16)/(factor-1)


    # does not work if rh=0 ??
    rh[(rh==0)] = 1e-12
    
    e_saturation = e_saturation_liquid(Tdry)*rh
    Tdew = dew_point(e_saturation)

    e0 = 6.11e2
    a =  17.269
    c = 35.86

    numerator = a*(c-T0)
    denominator = Tdew-c

    e_dash = e_saturation*(-numerator/denominator**2)
    e_dash_dash = e_saturation*((numerator/denominator**2)**2 + 2*numerator/denominator**3)

    Lv = 2.5e6
    Cp = 1004
    epsilon = 0.622

    f = p*Cp/(Lv*epsilon)

    A = e_dash_dash*0.5
    B = e_dash+f-Tdew*e_dash_dash
    C = -Tdry*f-Tdew*e_dash+0.5*Tdew**2*e_dash_dash

    Twet = (-B+np.sqrt(B*B-4*A*C))/(2*A)
    
    return Twet


def fetch_model(vrs, fields, alt_site, is35):    
    out = {}
    model_heights = vrs['height'][:] + alt_site # now above mean sea level
    # into ordinary numpy array ?
    model_heights = np.array(model_heights)

    model_time = vrs['time'][:]
    new_grid = np.mean(model_heights, axis=0)
    nx = model_time.shape[0]
    ny = new_grid.shape[0] 
    if is35:
        ind = 0
    else:
        ind = 1                    
    for field in fields:
        data = np.array(vrs[field][:])
        datai = np.zeros((nx, ny))
        if 'atten' in field:
            data = data[ind,:,:]
        # interpolate profiles into common altitude grid
        for n in range(0, len(model_time)):

            f = interp1d(model_heights[n,:], data[n,:], fill_value='extrapolate')
            datai[n,:] = f(new_grid)

        out[field] = datai
        
    return out, model_time, new_grid


def fetch_radar(vrs, fields, time):
    """ Read and rebin radar 2d fields in time"""
    out = {}
    x = vrs['time'][:]
    for field in fields:
        out[field] = rebin_x_2d(x, vrs[field][:], time)
    return out


def fetch_lidar(vrs, fields, time, height):
    """ Read and rebin lidar 2d fields in time and height"""
    out = {}        
    x = vrs['time'][:]
    lidar_alt = alt2m(vrs['altitude'])
    y = alt2m(vrs['range']) + lidar_alt    
    for field in fields:
        dataim = rebin_x_2d(x, vrs[field][:], time)
        dataim = rebin_x_2d(y, dataim.T, height).T
        out[field] = dataim
    return out


def alt2m(var):
    """ Read altitude variable (height, range, etc.) 
    and convert it to m if needed 
    """
    y = var[:]
    if var.units == 'km':
        y = y*1000
    return y


def interpolate_2d(x, y, z, xin, yin):
    """ FAST interpolation of 2d data that is in grid
    Does not work with nans! 
    """
    f = RectBivariateSpline(x, y, z, kx=1, ky=1) # linear interpolation
    return f(xin, yin)
    

def rebin_x_2d(x, data, xin):
    """ Rebin data in x-direction. Handles masked data. """
    # create new binning vector
    edge1 = round(xin[0] - (xin[1]-xin[0])/2)
    edge2 = round(xin[-1] + (xin[-1]-xin[-2])/2)
    edges = np.linspace(edge1, edge2, len(xin)+1)
    # prepare input/output data
    datai = np.zeros((len(xin), data.shape[1]))    
    data = ma.masked_invalid(data)    
    # loop over y
    for ii, values in enumerate(data.T):
        mask = values.mask
        if len(values[~mask])>0:
            datai[:,ii],_,_ = stats.binned_statistic(x[~mask], values[~mask], statistic='mean', bins=edges)

    datai[np.isfinite(datai)==0] = 0
    return ma.masked_equal(datai,0)    


def interpolate_lwp(time_lwp, lwp, time_new):
    """ Linear interpolation of LWP. This may not be good idea if we have lots of gaps in the data """
    try:
        f = interp1d(time_lwp, lwp)
        lwp_i = f(time_new)
    except:
        lwp_i = np.full_like(time_new, fill_value=np.nan)    
    return lwp_i
        

def get_lwp(mwr_vrs, frac_err, lin_err):
    """
    hatpro time can be 'hours since' 00h of measurement date
    or 'seconds since' some epoch (which could be site/file dependent)
    """
    try:
        lwp = mwr_vrs['LWP_data'][:]
        t = mwr_vrs['time'][:] 
        if (max(t) > 24): 
            t = epoch2desimal_hour((2001,1,1), t) # epoch is fixed here, but it
                                                  # might depend on the site/file!

            lwp_err = np.sqrt(lin_err**2 + (frac_err*lwp)**2) 

            return t, lwp, lwp_err
    except:
        return None, None, None # this is bad, fix later


def cat_Cnet_vars(vars_in, dvec, time_resolution):
    lin, log = 'linear', 'logarithmic'
    src = 'source'
    anc = 'ancillary_variables'
    bias_comm = 'This variable is an estimate of the one-standard-deviation calibration error'
    model_source = 'HYSPLIT'
    radar_source = 'Cloud radar model XXX'

    obs = []

    # general variables
    # -----------------
    var, lname = 'height', 'Height above mean sea level'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('height'), units='m', fill_value=None))

    var, lname = 'time', 'Time UTC'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('time'), units='hours since ' + dvec + ' 00:00:00 +0:00', 
                       fill_value=None, comment='Fixed ' + str(time_resolution) + 's resolution.'))

    var, lname = 'model_height', 'Height of model variables above mean sea level'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('model_height'), units='m', fill_value=None))

    var, lname = 'model_time', 'Model time UTC'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('model_time'), units='hours since ' + dvec + ' 00:00:00 +0:00', 
                       fill_value=None))

    var, lname = 'latitude', 'Latitude of site'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=(), units='degrees_north', fill_value=None))

    var, lname = 'longitude', 'Longitude of site'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=(), units='degrees_east', fill_value=None))

    var, lname = 'altitude', 'Altitude of site'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=(), units='m', fill_value=None, 
                       comment='Same as the altitude of radar or lidar - the one that is smaller (lower)'))

    # radar variables
    # ---------------
    var, lname = 'radar_frequency', 'Transmit frequency'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=(), units='GHz', fill_value=None))

    var, lname = 'Z', 'Radar reflectivity factor'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, units='dBZ', plot_range=(-40, 20), plot_scale=lin, 
                       extra_attributes={src:radar_source, anc: anc_names(var, True, True, True)}))

    var = 'Z_bias'
    obs.append(CnetVar(var, vars_in[var], long_name=bias_name(lname), size=(), units='dB', fill_value=None, comment=bias_comm))

    var = 'Z_error'
    obs.append(CnetVar(var, vars_in[var], long_name=err_name(lname), units='dB', comment='err'))
    
    var, lname = 'Z_sensitivity', 'Minimum detectable radar reflectivity'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('height'), units='dBZ',
                       comment=('This variable is an estimate of the radar sensitivity, i.e. the minimum detectable radar reflectivity\n',
                       'as a function of height. It includes the effect of ground clutter and gas attenuation but not liquid attenuation.')))
    
    var, lname = 'v', 'Doppler velocity'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, units='m s-1', plot_range=(-4, 2), plot_scale=lin, 
                       extra_attributes={src:radar_source}))

    var, lname = 'width', 'Spectral width'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, units='m s-1', plot_range=(0.03, 3), plot_scale=log, 
                       extra_attributes={src:radar_source}))
    
    var, lname = 'ldr', 'Linear depolarisation ratio'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, units='dB', plot_range=(-30, 0), plot_scale=lin, 
                       extra_attributes={src:radar_source}))

    # lidar variables
    # ---------------
    var, lname = 'lidar_wavelength', 'Laser wavelength'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=(), units='nm', fill_value=None))


    var, lname = 'beta', 'Attenuated backscatter coefficient'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, units='sr-1 m-1', plot_range=(1e-7, 1e-4), plot_scale=log, 
                       extra_attributes={src:'Lidar/Ceilometer model XXX', anc: anc_names(var, bias=True, err=True)}))

    var = 'beta_bias'
    obs.append(CnetVar(var, vars_in[var], long_name=bias_name(lname), size=(), units='dB', fill_value=None, comment=bias_comm))

    var = 'beta_error'
    obs.append(CnetVar(var, vars_in[var], long_name=err_name(lname), size=(), units='dB', fill_value=None,
                       comment='This variable is a crude estimate of the one-standard deviation random error'))
    
    # mwr variables
    # -------------
    var, lname = 'lwp', 'Liquid water path'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('time'), units='g m-2', plot_range=(-100, 1000), plot_scale=lin, 
                       extra_attributes={'source':'HATPRO microwave radiometer'}))

    var, lname = 'lwp_error', 'Error in liquid water path, one standard deviation'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('time'), units='g m-2'))
    
    # model variables
    # ---------------
    var, lname = 'temperature', 'Temperature'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('model_time','model_height'), units='K', plot_range=(200, 300), plot_scale=lin, 
                       extra_attributes={src:model_source}))

    var, lname = 'pressure', 'Pressure'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('model_time','model_height'), units='Pa', plot_range=(0, 1.1e5), plot_scale=log, 
                       extra_attributes={src:model_source}))

    var, lname = 'specific_humidity', 'Model specific humidity'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('model_time','model_height'), units='', plot_range=(0, 0.006), plot_scale=lin, 
                       extra_attributes={src:model_source}))

    var, lname = 'uwind', 'Zonal wind'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('model_time','model_height'), units='m s-1', plot_range=(-50, 50), plot_scale=lin, 
                       extra_attributes={src:model_source}))

    var, lname = 'vwind', 'Meridional wind'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, size=('model_time','model_height'), units='m s-1', plot_range=(-50, 50), plot_scale=lin, 
                       extra_attributes={src:model_source}))

    # other
    # -----
    var, lname = 'category_bits', 'Target classification bits'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, data_type='i4', units=None, fill_value=None,
                       extra_attributes={'valid_range':[0,5],
                                         'flag_masks':[0,1,2,3,4,5],
                                         'flag_meanings':'liquid_droplets falling_hydrometeors freezing_temperature melting_ice aerosols insects'}))

    var, lname = 'quality_bits', 'Data quality bits'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, data_type='i4', units=None, fill_value=None,
                       extra_attributes={'valid_range':[0,5],
                                         'flag_masks':[0,1,2,3,4,5],
                                         'flag_meanings':'lidar_echo radar_echo radar_clutter lidar_molec_scatter attenuation atten_correction'}))
    
    var, lname = 'Tw', 'Wet bulb temperature'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, units='K', fill_value=None, comment='Calculated from model T, P, and relative humidity, which were first interpolated into measurement grid.'))

    var, lname = 'insect_probability', 'Probability of insects'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, units='', fill_value=None, comment=''))

    var, lname = 'radar_gas_atten', 'Two-way radar attenuation due to atmospheric gases'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, units='dB', plot_range=(0,4), plot_scale=lin, comment=''))
    
    var, lname = 'radar_liquid_atten', 'Approximate two-way radar attenuation due to liquid water'
    obs.append(CnetVar(var, vars_in[var], long_name=lname, units='dB', plot_range=(0,4), plot_scale=lin, comment=''))
    
    return obs


def anc_names(var, bias=False, err=False, sens=False):
    out = ''
    if bias:
        out = out + var + '_bias '
    if err:
        out = out + var + '_error '
    if sens:
        out = out + var + '_sensitivity '
    out = out[:-1]
    return out


def get_site_location(f):
    lat = f['latitude'][:]
    lon = f['longitude'][:]
    return lat, lon


def get_time(reso):
    """ Fraction hour time vector 0-24 with "reso" [s] resolution """
    nsec = 24*60*60  
    step = 24.0/nsec*reso
    half_step = step/2.0
    time = np.arange(half_step, 24-half_step, step)
    return time


def get_height(alt_radar, alt_lidar, range_radar):
    """ Set radar measurement grid above mean sea level to our universal altitude grid.
    Site altitude is the altitude of lidar/radar depending which one is lower!
    """
    alt_site = min(alt_radar, alt_lidar)
    height = range_radar + alt_radar
    return height, alt_site


def epoch2desimal_hour(epoch, time_in):
    """ Convert seconds since epoch to desimal hour of that day
        IN:  epoch as tuple (year, month, day) 
        OUT: time as seconds since epoch 
    """
    dtime = []
    ep = calendar.timegm((epoch[0], epoch[1], epoch[2], 0, 0, 0))
    for t1 in time_in:
        x = time.gmtime(t1+ep)
        (h,m,s) = x[3:6]
        dt = h + ((m*60 + s)/3600)
        dtime.append(dt)
    # Last point can be 24h which would be 0 (we want 24 instead)
    if (dtime[-1] == 0):
        dtime[-1] = 24
    return dtime


def load_nc(file_in):
    f = netCDF4.Dataset(file_in)
    return f, f.variables


def save_cat(file_name, rad, time, height, model_time, model_height, obs, dvec, version, aux):
    rootgrp = netCDF4.Dataset(file_name, 'w', format='NETCDF4')
    # create dimensions
    time = rootgrp.createDimension('time', len(time))
    height = rootgrp.createDimension('height', len(height))
    model_time = rootgrp.createDimension('model_time', len(model_time))
    model_height = rootgrp.createDimension('model_height', len(model_height))
    
    # root group variables
    ncf.write_vars2nc(rootgrp, obs)

    # global attributes:
    rootgrp.Conventions = 'CF-1.7'
    rootgrp.title = 'Categorize file from ' + aux[0]
    rootgrp.institution = 'Data processed at the ' + aux[1]
    rootgrp.year = int(dvec[:4])
    rootgrp.month = int(dvec[5:7])
    rootgrp.day = int(dvec[8:])
    rootgrp.software_version = version
    rootgrp.git_version = ncf.git_version()
    rootgrp.file_uuid = str(uuid.uuid4().hex)
    rootgrp.references = 'https://doi.org/10.1175/BAMS-88-6-883'
    rootgrp.history = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S") + ' - categorize file created'
    rootgrp.close()



