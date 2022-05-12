import logging
import os
from bisect import bisect_left
from typing import Tuple, Union

import netCDF4
import numpy as np
from scipy.special import gamma

from cloudnetpy import utils
from cloudnetpy.datasource import DataSource
from cloudnetpy.products import product_tools
from cloudnetpy.products.product_tools import ProductClassification


class DrizzleSource(DataSource):
    """Class holding the input data for drizzle calculations.

    Args:
        categorize_file: Categorize file name.

    Attributes:
        mie (dict): Mie look-up table data.
        dheight (float): Median difference of height array.
        z (ndarray): 2D radar echo (linear units).
        beta (ndarray): 2D lidar backscatter.
        v (ndarray): 2D doppler velocity.

    """

    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.mie = self._read_mie_lut()
        self.dheight = utils.mdiff(self.getvar("height"))
        self.z = self._convert_z_units()
        self.beta = self.getvar("beta")
        self.v = self.getvar("v")

    def _convert_z_units(self):
        """Converts reflectivity factor to SI units."""
        z = self.getvar("Z") - 180
        z[z > 0.0] = 0.0
        return utils.db2lin(z)

    def _read_mie_lut(self):
        """Reads mie scattering look-up table."""
        mie_file = self._get_mie_file()
        with netCDF4.Dataset(mie_file) as nc:
            mie = nc.variables
            lut = {
                "Do": mie["lu_medianD"][:],
                "mu": mie["lu_u"][:],
                "S": mie["lu_k"][:],
                "lwf": mie["lu_LWF"][:],
                "termv": mie["lu_termv"][:],
            }
            band = self._get_wl_band()
            lut.update(
                {
                    "width": mie[f"lu_width_{band}"][:],
                    "ray": mie[f"lu_mie_ray_{band}"][:],
                    "v": mie[f"lu_v_{band}"][:],
                }
            )
        return lut

    @staticmethod
    def _get_mie_file():
        module_path = os.path.dirname(os.path.abspath(__file__))
        return "/".join((module_path, "mie_lu_tables.nc"))

    def _get_wl_band(self):
        """Returns string corresponding the radar frequency."""
        radar_frequency = float(self.getvar("radar_frequency"))
        wl_band = utils.get_wl_band(radar_frequency)
        return "35" if wl_band == 0 else "94"


class DrizzleClassification(ProductClassification):
    """Class storing the information about different drizzle types,
    child of  :class:`ProductClassification`.

    Args:
        categorize_file: Categorize file name.

    Attributes:
        is_v_sigma (ndarray): 2D array denoting finite v_sigma.
        warm_liquid (ndarray): 2D array denoting warm liquid.
        drizzle (ndarray): 2D array denoting drizzle presence.
        would_be_drizzle (ndarray): 2D array denoting possible drizzle pixels.
        cold_rain (ndarray): 1D array denoting profiles with melting layer.

    """

    def __init__(self, categorize_file: str):
        super().__init__(categorize_file)
        self.is_v_sigma = self._find_v_sigma(categorize_file)
        self.warm_liquid = self._find_warm_liquid()
        self.drizzle = self._find_drizzle()
        self.would_be_drizzle = self._find_would_be_drizzle()
        self.cold_rain = self._find_cold_rain()

    @staticmethod
    def _find_v_sigma(cat_file: str):
        v_sigma = product_tools.read_nc_fields(cat_file, "v_sigma")
        return np.isfinite(v_sigma)

    def _find_warm_liquid(self):
        return self.category_bits["droplet"] & ~self.category_bits["cold"]

    def _find_drizzle(self):
        return (
            ~utils.transpose(self.is_rain)
            & self.category_bits["falling"]
            & ~self.category_bits["droplet"]
            & ~self.category_bits["cold"]
            & ~self.category_bits["melting"]
            & ~self.category_bits["insect"]
            & self.quality_bits["radar"]
            & self.quality_bits["lidar"]
            & ~self.quality_bits["clutter"]
            & ~self.quality_bits["molecular"]
            & ~self.quality_bits["attenuated"]
            & self.is_v_sigma
        )

    def _find_would_be_drizzle(self):
        return (
            ~utils.transpose(self.is_rain)
            & self.warm_liquid
            & self.category_bits["falling"]
            & ~self.category_bits["melting"]
            & ~self.category_bits["insect"]
            & self.quality_bits["radar"]
            & ~self.quality_bits["clutter"]
            & ~self.quality_bits["molecular"]
        )

    def _find_cold_rain(self):
        return np.any(self.category_bits["melting"], axis=1)


class SpectralWidth:
    """Calculates corrected spectral width.

    Removes the effect of turbulence and horizontal wind that cause
    spectral broadening of the Doppler velocity.

    Args:
        categorize_file: Categorize file name.

    Attributes:
        categorize_file (str): Categorize file name.
        width_ht (ndarray): Spectral width containing the correction for turbulence broadening.

    """

    def __init__(self, categorize_file: str):
        self.cat_file = categorize_file
        self.width_ht = self._calculate_spectral_width()

    def _calculate_spectral_width(self):
        v_sigma = product_tools.read_nc_fields(self.cat_file, "v_sigma")
        try:
            width = product_tools.read_nc_fields(self.cat_file, "width")
        except KeyError:
            width = 0
            logging.warning(f"No spectral width, assuming width = {width}")
        sigma_factor = self._calc_v_sigma_factor()
        return width - sigma_factor * v_sigma

    def _calc_v_sigma_factor(self):
        beam_divergence = self._calc_beam_divergence()
        wind = self._calc_horizontal_wind()
        actual_wind = (wind + beam_divergence) ** (2 / 3)
        scaled_wind = (30 * wind + beam_divergence) ** (2 / 3)
        return actual_wind / (scaled_wind - actual_wind)

    def _calc_beam_divergence(self):
        beam_width = 0.5
        height = product_tools.read_nc_fields(self.cat_file, "height")
        return height * np.deg2rad(beam_width)

    def _calc_horizontal_wind(self):
        """Calculates magnitude of horizontal wind.

        Returns:
            ndarray: Horizontal wind (m s-1).

        """
        atmosphere = product_tools.interpolate_model(self.cat_file, ["uwind", "vwind"])
        u_wind = atmosphere["uwind"]
        v_wind = atmosphere["vwind"]
        return utils.l2norm(u_wind, v_wind)


class DrizzleSolver:
    """Estimates drizzle parameters.

    Args:
        drizzle_source: The :class:`DrizzleSource` instance.
        drizzle_class: The :class:`DrizzleClassification` instance.
        spectral_width: The :class:`SpectralWidth` instance.

    Attributes:
        params (dict): Dictionary of retrieved drizzle parameters 'Do', 'mu', 'S', 'beta_corr'.

    """

    def __init__(
        self,
        drizzle_source: DrizzleSource,
        drizzle_class: DrizzleClassification,
        spectral_width: SpectralWidth,
    ):
        self._data = drizzle_source
        self._drizzle_class = drizzle_class
        self._width_ht = spectral_width.width_ht
        self._width_lut = -self._data.mie["width"][:]
        self.params, self._dia_init = self._init_variables()
        self._beta_z_ratio = self._calc_beta_z_ratio()
        self._solve_drizzle(self._dia_init)

    def _init_variables(self) -> Tuple[dict, np.ndarray]:
        shape = self._data.z.shape
        res = {
            "Do": np.zeros(shape),
            "mu": np.zeros(shape),
            "S": np.zeros(shape),
            "beta_corr": np.ones(shape),
        }
        return res, np.zeros(shape)

    def _calc_beta_z_ratio(self) -> np.ndarray:
        return 2 / np.pi * self._data.beta / self._data.z

    def _find_lut_indices(self, ind, dia_init, n_dia, n_widths) -> Tuple[int, int]:
        ind_dia = bisect_left(self._data.mie["Do"], dia_init[ind], hi=n_dia - 1)
        ind_width = bisect_left(self._width_lut[:, ind_dia], -self._width_ht[ind], hi=n_widths - 1)
        return ind_width, ind_dia

    def _solve_drizzle(self, dia_init: np.ndarray):
        drizzle_ind = np.where(self._drizzle_class.drizzle == 1)
        dia_init[drizzle_ind] = self._calc_dia(self._beta_z_ratio[drizzle_ind], k=18.8)
        n_widths, n_dia = self._width_lut.shape[0], len(self._data.mie["Do"])
        max_ite = 10
        for ind in zip(*drizzle_ind):
            for _ in range(max_ite):
                lut_ind = self._find_lut_indices(ind, dia_init, n_dia, n_widths)
                dia = self._calc_dia(
                    self._beta_z_ratio[ind] * self.params["beta_corr"][ind],
                    self._data.mie["mu"][lut_ind[0]],
                    self._data.mie["ray"][lut_ind],
                    self._data.mie["S"][lut_ind],
                )
                self._update_result_tables(ind, dia, lut_ind)
                if self._is_converged(ind, dia, dia_init):
                    break
                self._dia_init[ind] = dia
            beta_factor = np.exp(
                2 * self.params["S"][ind] * self._data.beta[ind] * self._data.dheight
            )
            self.params["beta_corr"][ind[0], (ind[-1] + 1) :] *= beta_factor

    def _update_result_tables(self, ind: tuple, dia: Union[np.ndarray, float], lut_ind: tuple):
        self.params["Do"][ind] = dia
        self.params["mu"][ind] = self._data.mie["mu"][lut_ind[0]]
        self.params["S"][ind] = self._data.mie["S"][lut_ind]

    @staticmethod
    def _calc_dia(
        beta_z_ratio: Union[np.ndarray, float], mu: float = 0.0, ray: float = 1.0, k: float = 1.0
    ) -> Union[np.ndarray, float]:
        """Drizzle diameter calculation.

        Args:
            beta_z_ratio: Beta to z ratio, multiplied by (2 / pi).
            mu: Shape parameter for gamma calculations. Default is 0.
            ray: Mie to Rayleigh ratio for z. Default is 1.
            k: Alpha to beta ratio . Default is 1.

        Returns:
            ndarray: Drizzle diameter.

        References:
            https://journals.ametsoc.org/doi/pdf/10.1175/JAM-2181.1

        """
        const = ray * k * beta_z_ratio
        return (gamma(3 + mu) / gamma(7 + mu) * (3.67 + mu) ** 4 / const) ** (1 / 4)

    @staticmethod
    def _is_converged(ind: tuple, dia: Union[np.ndarray, float], dia_init: np.ndarray) -> bool:
        threshold = 1e-3
        return abs((dia - dia_init[ind]) / dia_init[ind]) < threshold
