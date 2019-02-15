import netCDF4
import numpy as np
from scipy.interpolate import interp1d
import cloudnetpy.products.ncf as ncf
from cloudnetpy.products.ncf import CnetVar

def generate_iwc(cat_file):
    cat = netCDF4.Dataset(cat_file) # open file
    vrs = cat.variables             # variables
    (_, nalt, _) = ncf.get_dimensions(cat)
    (T, meanT) = get_T(vrs)        
    ice_class = classificate_ice(vrs)
    (freq, is35) = ncf.get_radar_freq(vrs)
    spec_liq_atten = get_sla(is35)
    coeffs = get_iwc_coeffs(is35)
    iwc_bias = calc_iwc_bias(vrs, coeffs)    
    (rain_below_ice, rain_below_cold) = get_raining(vrs, ice_class['is_ice'], nalt)
    iwc_error = calc_iwc_error(vrs, coeffs, T, ice_class, spec_liq_atten, rain_below_ice)    
    (iwc, iwc_inc_rain) = calc_iwc(vrs['Z'][:], coeffs, T, ice_class['is_ice'], rain_below_ice)
    iwc_sensitivity = calc_iwc_sens(vrs['Z_sensitivity'][:], coeffs, meanT)
    retrieval_status = calc_iwc_status(iwc, ice_class, rain_below_ice, rain_below_cold)
    obs = lwc2cnet({'iwc_inc_rain': iwc_inc_rain, 'radar_frequency':freq, 'iwc':iwc, 'iwc_bias':iwc_bias, 'iwc_error':iwc_error, 'iwc_sensitivity':iwc_sensitivity, 'iwc_retrieval_status': retrieval_status})
    return (cat, obs) 


def lwc2cnet(vars_in):
    """ Defines IWC Cloudnet objects """
    log, lin = 'logarithmic', 'linear'
    obs = []

    freq = str(int(vars_in['radar_frequency']))
    
    s, lname = 'iwc', 'Ice water content'
    obs.append(CnetVar(s, vars_in[s], long_name=lname, units='kg m-3',
                       plot_scale=log, plot_range=(1e-7, 0.001), error_variable=True, bias_variable=True,
                       comment=("This variable was calculated from the " + freq + "-GHz radar reflectivity factor after correction for gaseous attenuation,\n",
                                "and temperature taken from a forecast model, using the following empirical formula:\n",
                                "log10(iwc[g m-3]) = 0.00058Z[dBZ]T[degC] + 0.0923Z[dBZ] + -0.00706T[degC] + -0.992.\n",
                                "In this formula Z is taken to be defined such that all frequencies of radar would measure the same Z in Rayleigh scattering ice.\n",
                                "However, the radar is more likely to have been calibrated such that all frequencies would measure the same Z in Rayleigh scattering\n",
                                "liquid cloud at 0 degrees C. The measured Z is therefore multiplied by |K(liquid,0degC,94GHz)|^2/0.93 = 0.7194 before applying this formula.\n",
                                "The formula has been used where the \"categorization\" data has diagnosed that the radar echo is due to ice, but note that in some cases\n",
                                "supercooled drizzle will erroneously be identified as ice. Missing data indicates either that ice cloud was present but it was only\n",
                                "detected by the lidar so its ice water content could not be estimated, or that there was rain below the ice associated with uncertain\n",
                                "attenuation of the reflectivities in the ice.\n",
                                "Note that where microwave radiometer liquid water path was available it was used to correct the radar for liquid attenuation when liquid\n",
                                "cloud occurred below the ice; this is indicated a value of 3 in the iwc_retrieval_status variable.  There is some uncertainty in this\n",
                                "prodedure which is reflected by an increase in the associated values in the iwc_error variable.\n",
                                "When microwave radiometer data were not available and liquid cloud occurred below the ice, the retrieval was still performed but its\n",
                                "reliability is questionable due to the uncorrected liquid water attenuation. This is indicated by a value of 2 in the iwc_retrieval_status\n",
                                "variable, and an increase in the value of the iwc_error variable"),
                       extra_attributes={'sensitivity_variable':'iwc_sensitivity'}))

    s = 'iwc_error'
    obs.append(CnetVar(s, vars_in[s], plot_scale=lin, plot_range=(0, 3), long_name=ncf.err_name(lname), units='dB',
                       comment=("This variable is an estimate of the one-standard-deviation random error in ice water content\n",
                                "due to both the uncertainty of the retrieval (about +50%/-33%, or 1.7 dB), and the random error in radar reflectivity factor from which ice water\n",
                                "content was calculated. When liquid water is present beneath the ice but no microwave radiometer data were available to correct for the\n",
                                "associated attenuation, the error also includes a contribution equivalent to approximately 250 g m-2 of liquid water path being uncorrected for.\n",
                                "As uncorrected liquid attenuation actually results in a systematic underestimate of ice water content, users may wish to reject affected data;\n",
                                "these pixels may be identified by a value of 2 in the iwc_retrieval_status variable.\n",
                                "Typical errors in temperature contribute much less to the overall uncertainty in retrieved ice water content so are not considered.\n",
                                "Missing data in iwc_error indicates either zero ice water content (for which an error in dB would be meaningless), or no ice water content value being reported.\n",
                                "Note that when zero ice water content is reported, it is possible that ice cloud was present but was just not detected by any of the instruments.")))

    s = 'iwc_bias'
    obs.append(CnetVar(s, vars_in[s], long_name=ncf.bias_name(lname), units='dB', size=(), fill_value=None, comment=ncf.bias_comm(lname)))
                           
    s = 'iwc_sensitivity'
    obs.append(CnetVar(s, vars_in[s], long_name="Minimum detectable ice water content", units='kg m-3', size=('height'),
                       comment=("This variable is an estimate of the minimum detectable ice water content as a function of height.")))

    s = 'iwc_retrieval_status'
    obs.append(CnetVar(s, vars_in[s], long_name=ncf.status_name(lname), units='', data_type='b', fill_value=None, plot_range=(0,7),
                       comment=("This variable describes whether a retrieval was performed for each pixel, and its associated quality, in the form of 8 different classes.\n",
                                "The classes are defined in the definition and long_definition attributes. The most reliable retrieval is that without any rain or liquid\n",
                                "cloud beneath, indicated by the value 1, then the next most reliable is when liquid water attenuation has been corrected using a microwave\n",
                                "radiometer, indicated by the value 3, while a value 2 indicates that liquid water cloud was present but microwave radiometer data were not\n",
                                "available so no correction was performed. No attempt is made to retrieve ice water content when rain is present below the ice; this is\n",
                                "indicated by the value 5."),
                       extra_attributes = {'definition':("0: No ice\n",
                                                         "1: Reliable retrieval\n",
                                                         "2: Unreliable: uncorrected attenuation\n",
                                                         "3: Retrieval with correction for liquid atten.\n",
                                                         "4: Ice detected only by the lidar\n",
                                                         "5: Ice above rain: no retrieval\n",
                                                         "6: Clear sky above rain\n",
                                                         "7: Would be identified as ice if below freezing"),
                                           'long_definition':("0: No ice present\n",
                                                              "1: Reliable retrieval\n",
                                                              "2: Unreliable retrieval due to uncorrected attenuation from liquid water below the ice (no liquid water path measurement available)\n",
                                                              "3: Retrieval performed but radar corrected for liquid attenuation using radiometer liquid water path which is not always accurate\n",
                                                              "4: Ice detected only by the lidar\n",
                                                              "5: Ice detected by radar but rain below so no retrieval performed due to very uncertain attenuation\n",
                                                              "6: Clear sky above rain, wet-bulb temperature less than 0degC: if rain attenuation were strong then ice could be present but undetected\n",
                                                              "7: Drizzle or rain that would have been classified as ice if the wet-bulb temperature were less than 0degC: may be ice if temperature is in error")}))

    s = 'iwc_inc_rain'
    obs.append(CnetVar(s, vars_in[s], long_name=lname, units='kg m-3',
                       plot_scale=log, plot_range=(1e-7, 0.001), error_variable='iwc_error', bias_variable='iwc_bias', 
                       comment=("This variable is the same as iwc, except that values of iwc in ice above rain have been included. \n",
                       "This variable contains values which have been severely affected by attenuation and should only be used when the effect of attenuation is being studied"),
                       extra_attributes={'sensitivity_variable':'iwc_sensitivity'}))
    
    return obs

            
def calc_iwc_status(iwc, ice_class, rain_below_cold, rain_below_ice):
    retrieved_ice = iwc > 0    
    retrieval_status = np.zeros((len(iwc),len(iwc[0])))
    retrieval_status[retrieved_ice] = 1
    retrieval_status[(retrieved_ice & ice_class['uncorrected_ice'])] = 2
    retrieval_status[(retrieved_ice & ice_class['corrected_ice'])] = 3
    retrieval_status[(~retrieved_ice & ice_class['is_ice'])] = 4
    retrieval_status[rain_below_cold] = 6
    retrieval_status[rain_below_ice] = 5    
    retrieval_status[(ice_class['would_be_ice'] & (retrieval_status == 0))] = 7
    return retrieval_status
    
def calc_iwc_sens(Z_sensitivity, coeffs, meanT):
    Z = Z_sensitivity + Z_scalefactor(coeffs['K2liquid0'])
    sensitivity = 10 ** (coeffs['cZT']*Z*meanT + coeffs['cT']*meanT + coeffs['cZ']*Z + coeffs['c']) * 0.001
    return sensitivity

def calc_iwc(Z, coeffs, T, is_ice, rain_below_ice):
    """ calculation of ice water content """
    Z = Z + Z_scalefactor(coeffs['K2liquid0'])
    iwc = 10 ** (coeffs['cZT']*Z*T + coeffs['cT']*T + coeffs['cZ']*Z + coeffs['c']) * 0.001
    iwc[~is_ice] = 0.0
    iwc_inc_rain = np.copy(iwc) 
    iwc[rain_below_ice] = np.nan
    return (iwc, iwc_inc_rain)
    
def get_raining(vrs, is_ice, nalt):
    """ True or False fields indicating raining below a) ice b) cold """
    a = (vrs['category_bits'][:] & 4) > 0
    rate = vrs['rainrate'][:] > 0  # True / False vector
    rate = ncf.expand_to_alt(rate, nalt)
    rain_below_ice = rate & is_ice
    rain_below_cold = rate & a
    return (rain_below_ice, rain_below_cold)

def classificate_ice(vrs):
    cb, qb = vrs['category_bits'][:], vrs['quality_bits'][:]
    # category bits:
    cb2  = (cb & 2)
    cb4  = (cb & 4)  
    cb8  = (cb & 8)  
    cb32 = (cb & 32)
    # quality bits:
    qb16 = (qb & 16)
    qb32 = (qb & 32)
    # do classification:
    is_ice = (cb2 > 0) & (cb4 > 0) & (cb8 == 0) & (cb32 == 0)
    would_be_ice = (cb2 > 0) & (cb4 == 0) & (cb32 == 0)
    corrected_ice = (qb16 > 0) & (qb32 > 0) & is_ice
    uncorrected_ice = (qb16 > 0) & (qb32 == 0) & is_ice
    return {'is_ice':is_ice, 'would_be_ice':would_be_ice, 'corrected_ice':corrected_ice, 'uncorrected_ice':uncorrected_ice}

def calc_iwc_bias(vrs, coeffs):
    return vrs['Z_bias'][:] * coeffs['cZ'] * 10

def calc_iwc_error(vrs, coeffs, T, ice_class, spec_liq_atten, rain_below_ice):
    MISSING_LWP = 250
    error = vrs['Z_error'][:] * (coeffs['cZT']*T + coeffs['cZ'])
    error = 1.7**2 + (error * 10)**2
    error[error > 0] = np.sqrt(error[error > 0])
    error[ice_class['uncorrected_ice']] = np.sqrt(1.7**2 + ((MISSING_LWP*0.001*2*spec_liq_atten)*coeffs['cZ']*10)**2)
    error[~ice_class['is_ice']] = np.nan
    error[rain_below_ice] = np.nan
    return error

def Z_scalefactor(K2liquid0):
    return 10 * np.log10(K2liquid0 / 0.93)

def get_sla(is35):
    """ specific liquid attenuation """
    if is35:
        return 1.0
    else:
        return 4.5
    
def get_T(vrs):
    """ linear interpolation of model temperature into target grid """
    f = interp1d(np.array(vrs['model_height'][:]), np.array(vrs['temperature'][:]))
    t_new = f(np.array(vrs['height'][:])) - 273.15
    t_mean = np.mean(t_new, axis=0)
    t_new[t_new > 0] = 0
    t_mean[t_mean > 0] = 0
    return (t_new, t_mean)

def get_iwc_coeffs(is35):
    if is35:
        a = 0.878
        b = 0.000242
        c = -0.0186
        d = 0.0699
        e = -1.63
    else:
        a = 0.669
        b = 0.000580
        c = -0.00706
        d = 0.0923
        e = -0.992
    return {'K2liquid0':a, 'cZT':b, 'cT':c, 'cZ':d, 'c':e}




    
