import numpy as np
from numpy import ma

import cloudnetpy.constants as con
from cloudnetpy import utils
from cloudnetpy.categorize import atmos_utils
from cloudnetpy.categorize.attenuations import Attenuation, calc_two_way_attenuation
from cloudnetpy.categorize.containers import ClassificationResult, Observations


class LiquidAttenuation:
    """Class for calculating liquid attenuation.

    References:
        Hogan, Robin & Connor, Ewan. (2004). Facilitating cloud radar and lidar
        algorithms: the Cloudnet Instrument Synergy/Target Categorization product.
    """

    def __init__(self, data: Observations, classification: ClassificationResult):
        self._model = data.model.data_dense
        self._liquid_in_pixel = classification.category_bits.droplet
        self._height = data.radar.height

        if data.mwr is not None:
            lwp = data.mwr.data["lwp"][:]
            lwp_error = data.mwr.data["lwp_error"][:]
        else:
            lwp = ma.masked_all(data.radar.time.size)
            lwp_error = ma.masked_all(data.radar.time.size)

        lwc_dz = atmos_utils.fill_clouds_with_lwc_dz(
            self._model["temperature"], self._model["pressure"], self._liquid_in_pixel
        )

        two_way_attenuation = self._calc_liquid_atten(lwp, lwc_dz)
        two_way_attenuation_error = self._calc_liquid_atten_err(lwp_error, lwc_dz)

        attenuated = utils.ffill(self._liquid_in_pixel)

        two_way_attenuation[~attenuated] = ma.masked
        two_way_attenuation_error[~attenuated] = ma.masked

        uncorrected = attenuated & two_way_attenuation.mask

        self.attenuation = Attenuation(
            amount=two_way_attenuation,
            error=two_way_attenuation_error,
            attenuated=attenuated,
            uncorrected=uncorrected,
        )

    def _calc_liquid_atten(
        self, lwp: ma.MaskedArray, lwc_dz: np.ndarray
    ) -> ma.MaskedArray:
        """Finds radar liquid attenuation."""
        lwp = lwp.copy()
        lwp[lwp < 0] = 0
        lwc_adiabatic = atmos_utils.calc_adiabatic_lwc(lwc_dz, self._height)
        lwc_scaled = atmos_utils.normalize_lwc_by_lwp(lwc_adiabatic, lwp, self._height)
        return self._calc_two_way_attenuation(lwc_scaled)

    def _calc_liquid_atten_err(
        self, lwp_error: ma.MaskedArray, lwc_dz: np.ndarray
    ) -> ma.MaskedArray:
        """Finds radar liquid attenuation error."""
        lwc_err_scaled = atmos_utils.normalize_lwc_by_lwp(
            lwc_dz, lwp_error, self._height
        )
        return self._calc_two_way_attenuation(lwc_err_scaled)

    def _calc_two_way_attenuation(self, lwc_scaled: np.ndarray) -> ma.MaskedArray:
        """Calculates liquid attenuation (dB).

        Args:
            lwc_scaled: Liquid water content (kg m-3).

        """
        specific_attenuation_rate = self._model["specific_liquid_atten"]
        specific_attenuation = specific_attenuation_rate * lwc_scaled * con.KG_TO_G
        return calc_two_way_attenuation(self._height, specific_attenuation)
