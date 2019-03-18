import numpy as np
from scipy.interpolate import interp1d
from cloudnetpy.categorize import DataSource
import cloudnetpy.utils as utils
import cloudnetpy.output as output


class DataCollecter(DataSource):
    def __init__(self, catfile):
        super().__init__(catfile)
        self.T, self.P = self.get_T_and_P()
        self.bits = self.get_bits()
        self.ptype = self.get_profile_type()
        self.dlwcdz = self.theory_adiabatic_lwc()
        self.lwp, self.lwp_error = self.get_lwp()
        self.size2d = (len(self.time), len(self.variables['height'][:]))

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


def generate_lwc(cat_file, output_file):
    data_handler = DataCollecter(cat_file)
    lwc = estimate_lwc(data_handler)
    redistribute_lwc(lwc, data_handler)


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
            lwc_th[ii, base:top + 1] = np.sum(lwc[ii, base:top + 1]) / (top - base + 1)

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
            lwc_adiabatic[ii, idx] = dlwc_dz * dheight * (np.arange(npoints) + 1)

            if np.any(data_handler.bits['radar_layer'][ii, idx]) or \
                    np.any(data_handler.bits['layer'][ii, top + 1:]):
                # good cloud boundaries
                retrieval_status[ii, idx] = 4
            else:
                # unknown cloud top; may need to adjust cloud top
                retrieval_status[ii, idx] = 5

        if data_handler.lwp[ii] > 0 and data_handler.ptype['is_melting_layer_in_profile'][ii]:
            lwc[ii, :] = lwc_adiabatic[ii, :]

            if data_handler.lwp[ii] > (np.sum(lwc[ii, :]) * dheight):
                index = np.where(retrieval_status[ii, :] == 5)[0]

                if len(index) > 0:
                    # Tänne tulee harvemmin, ei vielä kertaakaan
                    retrieval_status[ii, retrieval_status[ii, :] > 0] = 2
                    # index is now first cloud free pixel
                    index = index[-1]
                    ind = np.where(index == liquid_tops)[0]
                    index = index + 1
                    dlwc_dz = data_handler.theory_adiabatic_lwc(ii, liquid_bases[ind])
                    while (data_handler.lwp[ii] > (np.sum(lwc[ii, :]) * dheight)) \
                            and index <= data_handler.variables['height'][:]:
                        lwc[ii, index] = dheight * dlwc_dz * (index - liquid_bases[ind] + 1)
                        retrieval_status[ii, index] = 3
                        index = index + 1
                        lwc[ii, index - 1] = (data_handler.lwp[ii] - (np.sum(lwc[ii, 0:index - 1]) * dheight)) / dheight
                else:
                    # Välillä myös tänne
                    lwc[ii, :] = data_handler.lwp[ii] * lwc[ii, :] / (np.sum(lwc[ii, :]) * dheight)
                    retrieval_status[ii, retrieval_status[ii, :] > 2] = 1
            else:
                # Yleensä tulee tänne
                lwc[ii, :] = data_handler.lwp[ii] * lwc[ii, :] / (np.sum(lwc[ii, :]) * dheight)
                retrieval_status[ii, retrieval_status[ii, :] > 2] = 1

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
