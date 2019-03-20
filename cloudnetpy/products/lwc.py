import numpy as np
from scipy.interpolate import interp1d
from cloudnetpy.categorize import DataSource
import cloudnetpy.utils as utils
import cloudnetpy.output as output
import matplotlib.pyplot as plt

"""
Käydään vielä läpi, mitä kaikkea koodi touhuaa.
Nyt koodia on muokattu, muttei kovin tarkasti mietitty, miten toimii ja mitä tekee.

Koodi ajaa itsensä läpi ja kirjoittaa ainakin oikean suuntaisen metadatan.

Koodi on toistaiseksi uudelleen jäsennelty järkevän oloisesti, kunnes Simo
kumoaa kaiken.

Myös metadatan sisälty täytyy jossain vaiheessa miettiä, mitä sinne haluaa,
pätee tosin kaikkiin tuotteisiin, joten voinee tehdä kerralla, kun kaikki on valmista.

Kunhan nämä iwc ja iwc on vähän kauniimmat, voidaan ruveta pohtimaan plottaamista
"""

class DataCollecter(DataSource):
    def __init__(self, catfile):
        super().__init__(catfile)
        self.T, self.P = self.get_T_and_P()
        self.bits = self.get_bits()
        self.ptype = self.get_profile_type()
        self.dlwcdz = self.theory_adiabatic_lwc()
        self.lwp, self.lwp_error = self.get_lwp()
        self.size2d = (len(self.time),len(self.variables['height'][:]))


    def get_T_and_P(self):
        """ linear interpolation of model temperature into target grid """
        ftemp = interp1d(np.array(self.variables['model_height'][:]),
                         np.array(self.variables['temperature'][:]))
        fpres = interp1d(np.array(self.variables['model_height'][:]),
                         np.array(self.variables['pressure'][:]))
        return (ftemp(np.array(self.variables['height'][:])),
                fpres(np.array(self.variables['height'][:])))


    def get_bits(self):
        cb, qb = self.variables['category_bits'][:], \
                 self.variables['quality_bits'][:]
        bits = {}
        bits['droplet'] = utils.isbit(cb, 0)
        bits['melting'] = utils.isbit(cb, 3)
        bits['lidar'] = utils.isbit(qb, 1)
        bits['radar'] = utils.isbit(qb, 0)

        bits['layer'] = bits['lidar'] & bits['droplet']
        bits['radar_layer'] = bits['radar'] & bits['droplet']
        return bits


    def get_profile_type(self):
        # TODO: pitää tarkistaa toimiiko oikein ja mitä pitäisi tuottaa
        ptype = {}
        ptype['is_rain_in_profile'] = self.variables['rainrate'][:].astype(bool)
        ptype['is_melting_layer_in_profile'] = np.any(self.bits['melting'], axis=1)
        ptype['is_some_liquid'] = np.any(self.bits['droplet'], axis=1)
        ptype['is_rain'] = np.tile(ptype['is_rain_in_profile'],
                                   (len(self.variables['height'][:]), 1)).T
        return ptype


    def temperature2mixingratio(self, T, P):
        tt = 273.16
        t1 = (T / tt)
        t2 = 1 - (tt / T)
        svp = 10 ** (10.79574 * (t2) - 5.028 * np.log10(t1) + 1.50475e-4 * (
                    1 - (10 ** (-8.2969 * (t1 - 1)))) + 0.42873e-3 * (10 ** (4.76955 * (t2))) + 2.78614)
        mixingratio = 0.62198 * svp / (P - svp)
        return (mixingratio, svp)


    def theory_adiabatic_lwc(self, *args):
        """ Calculates the theoretical adiabatic rate of increase of LWC with
        height, or pressure, given the cloud base temperature and pressure
        Returns: dlwc/dz in kg m-3 m-1
        From Brenguier (1991) """
        if bool(args) == True:
            T = self.T[args[0], args[1]]
            P = self.P[args[0],  args[1]]
        else:
            T = self.T
            P = self.P

        e = 0.62198  # ratio of the molecular weight of water vapor to dry air
        g = -9.81  # acceleration due to gravity (m s-1)
        cp = 1005  # heat capacity of air at const pressure (J kg-1 K-1)
        L = 2.5e6  # latent heat of evaporation (J kg-1)
        R = 461.5 * e  # specific gas constant for dry air (J kg-1 K-1)
        drylapse = -g / cp  # dry lapse rate (K m-1)
        qs, es = self.temperature2mixingratio(T, P)
        rhoa = P / (R * (1 + 0.6 * qs) * T)
        dqldz = - (1 - (cp * T / (L * e))) * \
                (1 / ((cp * T / (L * e)) + (L * qs * rhoa / (P - es)))) \
                * (rhoa * g * e * es) * ((P - es) ** (-2))
        dlwcdz = rhoa * dqldz
        return dlwcdz


    def get_lwp(self):
        """ Read liquid water path (lwp) from a categorization file """
        if 'lwp' in self.variables:
            lwp = self.variables['lwp'][:]
            lwp[lwp < 0] = 0
            (lwp, lwp_error) = self.convert_lwp_units()
        else:
            lwp = np.zeros(self.time)
            lwp_error = np.zeros(self.time)

        #self.append_data(lwp,'LWP')
        self.append_data(lwp_error, 'LWP_error')
        return (lwp, lwp_error)


    def convert_lwp_units(self):
        """ Convert liquid water path (and its error) from [g m2] to [kg m2] """
        n = 1 if "kg" in self.variables['lwp'].units else 1000
        m = 1 if "kg" in self.variables['lwp_error'].units else 1000
        return (self.variables['lwp'][:] / n, self.variables['lwp_error'][:] / m)




def generate_lwc(cat_file, output_file):
    """Main function for generating liquid water content for Cloudnet.
    
    Args:
        cat_file: file name of categorize file.

    Returns: 
        - Pointer to (opened) categorize file.
        - ....
    """
    data_handler = DataCollecter(cat_file)

    # Mitäs hittoa, palauttetaan kaksi kertaa lwp?
    # Näkyy muokkaavan vain vanhaa jotenkin, ei siis ongelma
    #(lwp, lwp_error) = get_lwp(vrs, ntime)
    lwc = estimate_lwc(data_handler)
    calc_lwc_error(lwc, data_handler)
    redistribute_lwc(lwc, data_handler)
    """
    obs = lwc2cnet({'lwc_th':lwc_th,
                    'LWP':lwp,
                    'LWP_error':lwp_error,
                    'lwc':lwc,
                    'lwc_error':lwc_error,
                    'lwc_retrieval_status':retrieval_status})
    """
    output.update_attributes(data_handler.data)
    _save_data_and_meta(data_handler, output_file)


def calc_lwc_error(lwc, data_handler):
    lwc_error = np.zeros(data_handler.size2d)
    cloud_boundary_uncertainty = np.zeros(data_handler.size2d)
    lwp_error = data_handler.variables['lwp_error'][:]

    ind = np.where(~np.isnan(data_handler.lwp))  # to avoid dividing with nan
    lwp_error[ind] = lwp_error[ind] / data_handler.lwp[ind]
    lwp_uncertainty = np.tile(lwp_error, (len(data_handler.variables['height'][:]), 1)).T
    lwp_uncertainty[(~np.isnan(lwp_uncertainty)) & ((lwp_uncertainty < 0.1) | (lwp_uncertainty > 10))] = 10
    index = np.where(lwc != 0)

    cloud_boundary_uncertainty[index] = np.abs(np.gradient(lwc[index]))
    a = cloud_boundary_uncertainty[index]**2
    b = lwp_uncertainty[index]**2
    c = a + b

    ind = np.where(~np.isnan(c))
    c[ind] = np.sqrt(c[ind])

    lwc_error[index] = c
    lwc_error[(np.isnan(lwc)) | (data_handler.ptype['is_rain']) | (lwc == 0)] = np.nan

    data_handler.append_data(lwc_error, 'lwc_error')


def redistribute_lwc(lwc, data_handler):
    zero = np.zeros(1)
    lwc_th = np.zeros(data_handler.size2d)
    droplet_bit = np.zeros(data_handler.size2d)
    lwc[np.isnan(lwc)] = 0
    droplet_bit[lwc > 0] = 1  # is this always the same as ptype['is_some_liquid'] ??
    is_some_liquid = np.any(droplet_bit, axis=1)
    
    for ii in np.where(is_some_liquid)[0]:
        db = np.concatenate((zero, droplet_bit[ii, :], zero))
        db_diff = np.diff(db)
        liquid_bases = np.where(db_diff == 1)[0]
        liquid_tops = np.where(db_diff == -1)[0] - 1        
        for base, top in zip(liquid_bases, liquid_tops):
            lwc_th[ii, base:top+1] = np.sum(lwc[ii, base:top+1]) / (top-base + 1)

    data_handler.append_data(lwc_th, 'lwc_th')
            

def estimate_lwc(data_handler):
    # Alustetaan arrayt, joita muokataan
    zero = np.zeros(1)
    lwc = np.zeros(data_handler.size2d)
    lwc_adiabatic = np.zeros(data_handler.size2d)
    retrieval_status = np.zeros(data_handler.size2d)
    dheight = np.median(np.diff(data_handler.variables['height'][:]))

    # indeksit joissa 'is_some_liquid' == TRUE
    for ii in np.where(data_handler.ptype['is_some_liquid'])[0]:
        db = np.concatenate((zero, data_handler.bits['droplet'][ii, :], zero))
        db_diff = np.diff(db)
        liquid_bases = np.where(db_diff == 1)[0]
        liquid_tops = np.where(db_diff == -1)[0] - 1

        for base, top in zip(liquid_bases, liquid_tops):
            npoints = top - base + 1
            idx = np.arange(npoints) + base
            dlwc_dz = data_handler.theory_adiabatic_lwc(ii, base)  # constant
            lwc_adiabatic[ii, idx] = dlwc_dz * dheight * (np.arange(npoints)+1)

            if np.any(data_handler.bits['radar_layer'][ii, idx]) or \
                    np.any(data_handler.bits['layer'][ii, top+1:]):
                # good cloud boundaries
                retrieval_status[ii, idx] = 4
            else:
                # unknown cloud top; may need to adjust cloud top
                retrieval_status[ii, idx] = 5

        if data_handler.lwp[ii] > 0 and data_handler.ptype['is_melting_layer_in_profile'][ii]:
            lwc[ii, :] = lwc_adiabatic[ii, :]


            plt.plot(ii,data_handler.lwp[ii],'b.')
            plt.plot(ii,(np.sum(lwc[ii, :]) * dheight),'r.')

            if data_handler.lwp[ii] > (np.sum(lwc[ii, :]) * dheight):

                index = np.where(retrieval_status[ii, :] == 5)[0]

                if len(index) > 0:

                    print('jee')
                    # Tänne tulee harvemmin, ei vielä kertaakaan
                    retrieval_status[ii, retrieval_status[ii, :] > 0] = 2
                    # index is now first cloud free pixel
                    index = index[-1]
                    ind = np.where(index == liquid_tops)[0]
                    index = index + 1
                    dlwc_dz = data_handler.theory_adiabatic_lwc(ii, liquid_bases[ind])
                    while (data_handler.lwp[ii] > (np.sum(lwc[ii, :])*dheight)) \
                            and index <= data_handler.variables['height'][:]:
                        lwc[ii, index] = dheight * dlwc_dz * (index-liquid_bases[ind]+1)
                        retrieval_status[ii, index] = 3
                        index = index + 1
                        lwc[ii, index-1] = (data_handler.lwp[ii] - (np.sum(lwc[ii, 0:index-1]) * dheight)) / dheight
                else:
                    # Välillä myös tänne
                    print('jees')
                    lwc[ii, :] = data_handler.lwp[ii] * lwc[ii, :] / (np.sum(lwc[ii, :]) * dheight)
                    retrieval_status[ii, retrieval_status[ii, :] > 2] = 1
            else:
                # Yleensä tulee tänne
                lwc[ii, :] = data_handler.lwp[ii] * lwc[ii, :] / (np.sum(lwc[ii, :]) * dheight)
                retrieval_status[ii, retrieval_status[ii, :] > 2] = 1

    plt.show()

    # some additional screening..
    retrieval_status[data_handler.ptype['is_rain']] = 6
    lwc[data_handler.ptype['is_rain']] = np.nan
    lwc[(retrieval_status == 4) | (retrieval_status == 5)] = np.nan
    data_handler.lwp[(data_handler.ptype['is_rain_in_profile']) |
                     (data_handler.lwp == 0) |
                     (data_handler.ptype['is_melting_layer_in_profile'])] = np.nan

    data_handler.append_data(lwc, 'lwc')
    data_handler.append_data(retrieval_status, 'lwc_retrieval_status')
    
    return lwc


def _save_data_and_meta(data_handler, output_file):
    """
    Saves wanted information to NetCDF file.
    """
    dims = {'time': len(data_handler.time),
            'height': len(data_handler.variables['height'])}
    rootgrp = output.init_file(output_file, dims, data_handler.data, zlib=True)
    vars_from_source = ('altitude', 'latitude', 'longitude', 'time', 'height')
    output.copy_variables(data_handler.dataset, rootgrp, vars_from_source)
    rootgrp.title = f"Classification file from {data_handler.dataset.location}"
    rootgrp.source = f"Categorize file: {_get_source(data_handler)}"
    output.copy_global(data_handler.dataset, rootgrp, ('location', 'day',
                                                       'month', 'year'))
    output.merge_history(rootgrp, 'classification', data_handler)
    rootgrp.close()


def _get_source(data_handler):
    """Returns uuid (or filename if uuid not found) of the source file."""
    return getattr(data_handler.dataset, 'file_uuid', data_handler.filename)
