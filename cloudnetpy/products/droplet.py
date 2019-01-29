import numpy as np
import scipy.signal
#import matplotlib.pyplot as plt
import sys
import ncf 

def estimate_v_sigma(v):
    """ No good method for this yet """
    v_sigma = abs(0.01 * v)
    return v_sigma
    

def get_rain_bit(Z, time):
    MIN_TIME = 5 # minutes
    rain_bit = np.zeros((Z.shape[0]), dtype=int)
    rain_bit[Z[:,3]>0] = 1
    step = ncf.med_diff(time)*60*60 # minutes
    nsteps = int(round(MIN_TIME*60/step/2))
    for ind in np.where(rain_bit)[0]:
        i1 = max(0, ind-nsteps)
        i2 = min(ind+nsteps+1, len(rain_bit))
        rain_bit[i1:i2] = 1
    return rain_bit
    

def get_radar_clutter(rain_bit, v, v_sigma):
    """ Estimate clutter from radar data. """
    NGATES = 10
    VLIM = 0.05

    clutter_bit = np.zeros(v.shape, dtype=int)

    no_rain = np.where(rain_bit == 0)[0]
    ind = np.ma.where(np.abs(v[no_rain, 0:NGATES])<VLIM)

    for n,m in zip(*ind):
        clutter_bit[no_rain[n],m] = 1
        
    return clutter_bit


def correct_liquid_cloud_top(Z, beta, Tw, cold_bit, cloud_bit, cloud_top, height):
    """ Correct lidar detected cloud top using radar signal.
    THIS NEEDS WORK!!
    """
    
    dheight = ncf.med_diff(height) 
    top_above = int(np.ceil((300/dheight)))
    
    ind = np.where(cloud_top)

    for n,t in zip(*ind):
        
        if (cold_bit[n,t]):
            ii = top_above
        else:
            ii = np.where(cold_bit[n,t:]==1)[0][0]
            
        rad = Z[n,t:t+ii]
        alt = height[t:t+ii]
            
        if (rad.mask.all()):
            pass

        elif (not rad.mask.any()):
            pass
            #cloud_bit[n,t:t+ii] = 1

        else:
            #plt.plot(rad,alt,'ro-')            
            iii = np.ma.nonzero(rad)[0][-1]
            #plt.plot(rad[iii],alt[iii],'go',markersize=10)
            #plt.show()            
            cloud_bit[n,t:t+iii+1] = 1
        
    ind = np.where(Tw<273.16-40)
    cloud_bit[ind] = 0
            
    return cloud_bit


def get_base_ind(dprof, p, dist, lim):
    start = max(p-dist, 0) # should not be negative 
    diffs = dprof[start:p]
    mind = np.argmax(diffs)
    return start + np.where(diffs > diffs[mind]/lim)[0][0]

        
def get_top_ind(dprof, p, nprof, dist, lim):
    end = min(p+dist, nprof) # should not be greater than len(profile)
    diffs = dprof[p:end]
    mind = np.argmin(diffs)
    return p + np.where(diffs < diffs[mind]/lim)[0][-1] + 1


def get_liquid_layers(beta, height):
    """
    Estimate liquid layers from lidar backscattering data.
    """

    # parameters for the peak shape
    PEAK_AMP = 2e-5
    MAX_WIDTH = 300 
    MIN_POINTS = 3
    MIN_TOP_DER = 4e-7
    
    # search distances for potential base/top
    dheight = ncf.med_diff(height) 
    base_below_peak = int(np.ceil((200/dheight)))
    top_above_peak = int(np.ceil((150/dheight)))
    
    # init result matrices (maybe better be masked arrays)
    cloud_bit = np.zeros(beta.shape, dtype=int)
    base_bit = np.zeros(beta.shape, dtype=int)
    top_bit = np.zeros(beta.shape, dtype=int)
    
    # set missing values to 0
    beta_diff = np.diff(beta,axis=1).filled(fill_value=0) # difference matrix
    beta = beta.filled(fill_value=0)

    # all peaks
    pind = scipy.signal.argrelextrema(beta, np.greater, order=4, axis=1)

    # take strong peaks only
    strong_peaks = np.where(beta[pind] > PEAK_AMP)
    pind = (pind[0][strong_peaks], pind[1][strong_peaks])

    # loop over strong peaks
    for n,p in zip(*pind):

        lprof = beta[n,:]
        dprof = beta_diff[n,:]
        
        try:
            base = get_base_ind(dprof, p, base_below_peak, 4)
        except:
            continue

        try:
            top = get_top_ind(dprof, p, height.shape[0], top_above_peak, 4)
        except:
            continue

    
        tval, pval = lprof[top], lprof[p]

        # calculate peak properties
        npoints = np.count_nonzero(lprof[base:top+1])
        peak_width = height[top] - height[base]
        topder = (pval - tval) / peak_width

        if (npoints > MIN_POINTS and peak_width < MAX_WIDTH and topder > MIN_TOP_DER):            
            cloud_bit[n, base:top+1] = 1
            base_bit[n, base] = 1
            top_bit[n, top] = 1

    return cloud_bit, base_bit, top_bit


def get_T0_alt(Tw, height, T0):
    n = Tw.shape[0]
    alt = np.zeros((n,1))
    for ii, prof in enumerate(Tw):
        try:
            # x need to be increasing in python interpolation
            ind = np.where(prof<T0)[0][0]
            x = prof[ind-1:ind+1]
            y = height[ind-1:ind+1]
            if (x[0] > x[1]):
                x = np.flip(x, 0)
                y = np.flip(y, 0)
                
            alt[ii] = np.interp(T0, x, y)
        except:
            continue
    return alt




