import netCDF4
import numpy as np
from scipy.interpolate import interp1d
import ncf
from ncf import CnetVar

def generate_lwc(cat_file):
    """Main function for generating liquid water content for Cloudnet.
    
    Args:
        cat_file: file name of categorize file.

    Returns: 
        - Pointer to (opened) categorize file.
        - ....
    """
    cat = netCDF4.Dataset(cat_file) 
    vrs = cat.variables             
    (ntime, nalt, _) = ncf.get_dimensions(cat)
    (T, P) = get_T_and_P(vrs)
    bits = get_bits(vrs)
    ptype = get_profile_type(vrs, bits, nalt)
    (lwp, lwp_error) = get_lwp(vrs, ntime)    
    (lwc, lwp, retrieval_status) = estimate_lwc(bits, ptype, lwp, lwp_error, T, P, ntime, nalt, vrs)
    lwc_error = calc_lwc_error(lwc, lwp, ntime, nalt, vrs, ptype)
    lwc_th = redistribute_lwc(lwc, ntime, nalt)    
    obs = lwc2cnet({'lwc_th':lwc_th, 'lwp':lwp, 'lwp_error':lwp_error, 'lwc':lwc, 'lwc_error':lwc_error, 'lwc_retrieval_status':retrieval_status})
    return (cat, obs)
    

def lwc2cnet(vars_in):
    """ Defines LWC Cloudnet objects """
    log, lin = 'logarithmic', 'linear'
    obs = []
    
    s, lname = 'lwc', 'Liquid water content'
    obs.append(CnetVar(s, vars_in[s], long_name=lname, units='kg m-3',
                       plot_scale=log, plot_range=(1e-5, 0.01), error_variable=True,
                       comment=("This variable was calculated for the profiles where the \"categorization\" data has diagnosed that liquid water is present \n",
                                "and liquid water path is available from a coincident microwave radiometer. The model temperature and pressure were used to estimate the \n",
                                "theoretical adiabatic liquid water content gradient for each cloud base and the adiabatic liquid water content is then scaled so that its \n",
                                "integral matches the radiometer measurement so that the liquid water content now follows a quasi-adiabatic profile. If the liquid layer is \n",
                                "detected by the lidar only, there is the potential for cloud top height to be underestimated and so if the adiabatic integrated liquid water \n",
                                "content is less than that measured by the microwave radiometer, the cloud top is extended until the adiabatic integrated liquid water content \n",
                                "agrees with the value measured by the microwave radiometer. \n",
                                "Missing values indicate that either 1) a liquid water layer was diagnosed but no microwave radiometer data was available, \n",
                                "2) a liquid water layer was diagnosed but the microwave radiometer data was unreliable; this may be because a melting layer was present in the profile, \n",
                                "or because the retrieved lwp was unphysical (values of zero are not uncommon for thin supercooled liquid layers), or \n",
                                "3) that rain is present in the profile and therefore, the vertical extent of liquid layers is difficult to ascertain. ")))

    s = 'lwc_error'
    obs.append(CnetVar(s, vars_in[s], long_name=ncf.err_name(lname), units='kg m-3',
                       comment=("This variable is an estimate of the random error in liquid water content due to the uncertainty in the microwave radiometer \n",
                                "liquid water path retrieval and the uncertainty in cloud base and/or cloud top height. This is associated with the resolution of the grid used, 20 m, \n",
                                "which can affect both cloud base and cloud top. If the liquid layer is detected by the lidar only, there is the potential for cloud top height to be underestimated. \n",
                                "Similarly, there is the possibility that the lidar may not detect the second cloud base when multiple layers are present and the cloud base will be \n",
                                "overestimated. It is assumed that the error contribution arising from using the model temperature and pressure at cloud base is negligible.")))
    
    s = 'lwc_retrieval_status'
    obs.append(CnetVar(s, vars_in[s], long_name='Liquid water content retrieval status', units='', data_type='b', fill_value=None, plot_range=(0,6),
                       comment=("This variable describes whether a retrieval was performed for each pixel, and its associated quality, in the form of 6 different classes.\n",
                                "The classes are defined in the definition and long_definition attributes. The most reliable retrieval is that when both radar and lidar detect the liquid layer, and \n",
                                "microwave radiometer data is present, indicated by the value 1. The next most reliable is when microwave radiometer data is used to adjust the cloud depth when \n",
                                "the radar does not detect the liquid layer, indicated by the value 2, with a value of 3 indicating the cloud pixels that have been added at cloud top to avoid the \n",
                                "profile becoming superadiabatic. A value of 4 indicates that microwave radiometer data were not available or not reliable (melting level present or unphysical values) \n",
                                "but the liquid layers were well defined.  If cloud top was not well defined then this is indicated by a value of 5. \n",
                                "The full retrieval of liquid water content, which requires reliable liquid water path from the microwave radiometer, was only performed for classes 1-3. \n",
                                "No attempt is made to retrieve liquid water content when rain is present; this is indicated by the value 6."),
                       extra_attributes = {'definition':("0: No liquid water\n",
                                                         "1: Reliable retrieval\n",
                                                         "2: Adiabatic retrieval: cloud top adjusted\n",
                                                         "3: Adiabatic retrieval: new cloud pixel\n",
                                                         "4: Unreliable lwp: no retrieval\n",
                                                         "5: Unreliable lwp/cloud boundaries: no retrieval\n",
                                                         "6: Rain present: no retrieval"), 
                                           'long_definition':("0: No liquid water detected\n",
                                                              "1: Reliable retrieval \n",
                                                              "2: Adiabatic retrieval where cloud top has been adjusted to match liquid water path from microwave radiometer because layer is not detected by radar\n",
                                                              "3: Adiabatic retrieval: new cloud pixels where cloud top has been adjusted to match liquid water path from microwave radiometer because layer is not detected by radar\n",
                                                              "4: No retrieval: either no liquid water path is available or liquid water path is uncertain\n",
                                                              "5: No retrieval: liquid water layer detected only by the lidar and liquid water path is unavailable or uncertain: cloud top may be higher than diagnosed cloud top since lidar signal has been attenuated\n",
                                                              "6: Rain present: cloud extent is difficult to ascertain and liquid water path also uncertain.")}))    

    s, lname = 'lwp',  'Liquid water path'
    obs.append(CnetVar(s, vars_in[s], long_name=lname, units='kg m-2',
                       plot_scale=lin, plot_range=(-100, 1000), size=('time'), error_variable=True,
                       comment=("This variable is the vertically integrated liquid water directly over the site.\n",
                                "The temporal correlation of errors in liquid water path means that it is not really meaningful to distinguish bias from random error,\n",
                                "so only an error variable is provided.\n",
                                "Original comment: These values denote the vertically integrated amount of condensed water from the surface to TOA.")))    
    s = 'lwp_error'
    obs.append(CnetVar(s, vars_in[s], long_name=ncf.err_name(lname), units='kg m-2', size=('time'),
                       comment=('This variable is a rough estimate of the one-standard-deviation error in liquid water path, calculated as a',
                                'combination of a 20 g m-2 linear error and a 25% fractional error.')))
    
    s = 'lwc_th'
    obs.append(CnetVar(s, vars_in[s], long_name='Liquid water content (tophat distribution)', units='kg m-3',
                       comment=("This variable is the liquid water content assuming a tophat distribution.",
                                "I.e. the profile of liquid water content in each layer is constant.")))

    return obs


def calc_lwc_error(lwc, lwp, ntime, nalt, vrs, ptype):
    lwc_error = np.zeros((ntime, nalt))
    cloud_boundary_uncertainty = np.zeros((ntime, nalt))
    lwp_error = vrs['lwp_error'][:]
    ind = np.where(~np.isnan(lwp)) # to avoid dividing with nan
    lwp_error[ind] = lwp_error[ind] / lwp[ind]
    lwp_uncertainty = ncf.expand_to_alt(lwp_error, nalt)
    lwp_uncertainty[(~np.isnan(lwp_uncertainty)) & ((lwp_uncertainty < 0.1) | (lwp_uncertainty > 10))] = 10
    index = np.where(lwc != 0)
    cloud_boundary_uncertainty[index] = np.abs(np.gradient(lwc[index]))
    a = cloud_boundary_uncertainty[index]**2
    b = lwp_uncertainty[index]**2
    c = a + b
    ind = np.where(~np.isnan(c))
    c[ind] = np.sqrt(c[ind])
    lwc_error[index] = c
    lwc_error[(np.isnan(lwc)) | (ptype['is_rain']) | (lwc == 0)] = np.nan
    return lwc_error


def redistribute_lwc(lwc, ntime, nalt):
    zero = np.zeros(1)
    lwc_th = np.zeros((ntime, nalt))
    droplet_bit = np.zeros((ntime, nalt))
    lwc[np.isnan(lwc)] = 0
    droplet_bit[lwc > 0] = 1 # is this always the same as ptype['is_some_liquid'] ??
    is_some_liquid = np.any(droplet_bit, axis=1)
    
    for ii in np.where(is_some_liquid)[0]:
        db = np.concatenate((zero, droplet_bit[ii,:], zero))
        db_diff = np.diff(db)
        liquid_bases = np.where(db_diff == 1)[0]
        liquid_tops = np.where(db_diff == -1)[0] - 1        
        for base, top in zip(liquid_bases, liquid_tops):
            lwc_th[ii,base:top+1] = np.sum(lwc[ii,base:top+1]) / (top-base + 1)
    return lwc_th
            

def estimate_lwc(bits, ptype, lwp, lwp_error, T, P, ntime, nalt, vrs):    
    zero = np.zeros(1)
    lwc = np.zeros((ntime, nalt))
    lwc_adiabatic = np.zeros((ntime, nalt))
    retrieval_status = np.zeros((ntime, nalt))
    dheight = np.median(np.diff(vrs['height'][:]))

    for ii in np.where(ptype['is_some_liquid'])[0]:

        db = np.concatenate((zero, bits['droplet_bit'][ii,:], zero))
        db_diff = np.diff(db)
        liquid_bases = np.where(db_diff == 1)[0]
        liquid_tops = np.where(db_diff == -1)[0] - 1

        for base, top in zip(liquid_bases, liquid_tops):

            npoints = top - base + 1
            idx = np.arange(npoints) + base
            dlwc_dz = theory_adiabatic_lwc(T[ii,base], P[ii,base]) # constant
            lwc_adiabatic[ii,idx] = dlwc_dz * dheight * (np.arange(npoints)+1)

            if np.any(bits['radar_layer_bit'][ii,idx]) or np.any(bits['layer_bit'][ii,top+1:]):
                # good cloud boundaries
                retrieval_status[ii,idx] = 4
            else:
                # unknown cloud top; may need to adjust cloud top
                retrieval_status[ii,idx] = 5

        if (lwp[ii] > 0 and ~ptype['is_melting_layer_in_profile'][ii]):

            lwc[ii,:] = lwc_adiabatic[ii,:]
            
            if lwp[ii] > (np.sum(lwc[ii,:]) * dheight):
                
                index = np.where(retrieval_status[ii,:] == 5)[0]

                if len(index) > 0:

                    retrieval_status[ii,retrieval_status[ii,:] > 0] = 2
                    # index is now first cloud free pixel
                    index = index[-1]
                    ind = np.where(index == liquid_tops)[0]
                    index = index + 1
                    dlwc_dz = theory_adiabatic_lwc(T[ii,liquid_bases[ind]], P[ii,liquid_bases[ind]])
                    while (lwp[ii] > (np.sum(lwc[ii,:])*dheight)) and index <= nalt:
                        lwc[ii,index] = dheight * dlwc_dz * (index-liquid_bases[ind]+1)
                        retrieval_status[ii,index] = 3
                        index = index + 1
                        lwc[ii,index-1] = (lwp[ii] - (np.sum(lwc[ii,0:index-1]) * dheight)) / dheight
                else:
                    
                    lwc[ii,:] = lwp[ii] * lwc[ii,:] / (np.sum(lwc[ii,:]) * dheight)
                    retrieval_status[ii,retrieval_status[ii,:] > 2] = 1
            else:
                
                lwc[ii,:] = lwp[ii] * lwc[ii,:] / (np.sum(lwc[ii,:]) * dheight)
                retrieval_status[ii,retrieval_status[ii,:] > 2] = 1

    # some additional screening..
    retrieval_status[ptype['is_rain']] = 6
    lwc[ptype['is_rain']] = np.nan
    lwc[(retrieval_status == 4) | (retrieval_status == 5) ] = np.nan
    lwp[(ptype['is_rain_in_profile']) | (lwp == 0) | (ptype['is_melting_layer_in_profile'])] = np.nan
    
    return (lwc, lwp, retrieval_status)


def temperature2mixingratio(T, P):
    tt = 273.16
    t1 = (T/tt)
    t2 = 1 - (tt/T)
    svp = 10 ** (10.79574*(t2) - 5.028*np.log10(t1) + 1.50475e-4*(1-(10**(-8.2969*(t1-1)))) + 0.42873e-3*(10**(4.76955*(t2))) + 2.78614)
    mixingratio = 0.62198 * svp / (P-svp)
    return (mixingratio, svp)


def theory_adiabatic_lwc(T, P):
    """ Calculates the theoretical adiabatic rate of increase of LWC with
    height, or pressure, given the cloud base temperature and pressure
    Returns: dlwc/dz in kg m-3 m-1
    From Brenguier (1991) """
    e  = 0.62198       # ratio of the molecular weight of water vapor to dry air
    g  = -9.81         # acceleration due to gravity (m s-1)
    cp = 1005          # heat capacity of air at const pressure (J kg-1 K-1) 
    L  = 2.5e6         # latent heat of evaporation (J kg-1)
    R  = 461.5 * e     # specific gas constant for dry air (J kg-1 K-1)
    drylapse = -g / cp # dry lapse rate (K m-1)
    (qs, es) = temperature2mixingratio(T, P)
    rhoa = P / (R * (1 + 0.6*qs) * T)
    dqldz = - (1 - (cp*T / (L*e))) * ( 1 / ( (cp*T / (L*e)) + (L*qs*rhoa / (P-es)) )) * (rhoa*g*e*es) * ((P-es)**(-2))
    dlwcdz = rhoa * dqldz
    return dlwcdz


def get_lwp(vrs, ntime):
    """ Read liquid water path (lwp) from a categorization file """
    if 'lwp' in vrs:
        lwp = vrs['lwp'][:]
        lwp[lwp<0] = 0
        (lwp, lwp_error) = convert_lwp_units(vrs)
    else:
        lwp = np.zeros(ntime)
        lwp_error = np.zeros(ntime)
    return (lwp, lwp_error)


def convert_lwp_units(vrs):
    """ Convert liquid water path (and its error) from [g m2] to [kg m2] """
    n = 1 if "kg" in vrs['lwp'].units else 1000
    m = 1 if "kg" in vrs['lwp_error'].units else 1000
    return (vrs['lwp'][:] / n, vrs['lwp_error'][:] / m)


def get_profile_type(vrs, bits, nalt):
    ptype = {}
    ptype['is_rain_in_profile'] = vrs['rainrate'][:].astype(bool)
    ptype['is_melting_layer_in_profile'] = np.any(bits['melting_bit'], axis=1)    
    ptype['is_some_liquid'] = np.any(bits['droplet_bit'], axis=1)
    ptype['is_rain'] = ncf.expand_to_alt(ptype['is_rain_in_profile'], nalt)
    return ptype


def get_bits(vrs):
    cb, qb = vrs['category_bits'][:], vrs['quality_bits'][:]    
    bits = {}
    bits['droplet_bit'] = (cb & 1) > 0
    bits['melting_bit'] = (cb & 8) > 0
    bits['lidar_bit']   = (qb & 2) > 0
    bits['radar_bit']   = (qb & 1) > 0
    bits['layer_bit']       = bits['lidar_bit'] & bits['droplet_bit']
    bits['radar_layer_bit'] = bits['radar_bit'] & bits['droplet_bit']
    return bits


def get_T_and_P(vrs):
    """ linear interpolation of model temperature into target grid """
    ftemp = interp1d(np.array(vrs['model_height'][:]), np.array(vrs['temperature'][:]))
    fpres = interp1d(np.array(vrs['model_height'][:]), np.array(vrs['pressure'][:]))
    return (ftemp(np.array(vrs['height'][:])), fpres(np.array(vrs['height'][:])))


    

