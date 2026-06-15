import logging
from datetime import datetime, timezone
from os import PathLike

import numpy as np
import numpy.typing as npt
from numpy import ma

from cloudnetpy.datasource import DataSource
from cloudnetpy.products.product_tools import CategorizeBits, CategoryBits


class ObservationManager(DataSource):
    """Class to collect and manage observations for downsampling.

    Args:
        obs (str): Name of observation product
        obs_file (str): Path to source observation file

    Notes:
        Output is ObservationManager object where all product data and
        information is included.

        Class inherits DataSource interface from CloudnetPy. Observation file
        should be processed using CloudnetPy for this class to work properly.
    """

    def __init__(self, obs: str, obs_file: str | PathLike) -> None:
        super().__init__(obs_file)
        self.obs = obs
        self._file = obs_file
        self.date = self._get_date()
        self.radar_freq = self._get_radar_frequency()
        self.z_sensitivity = self._get_z_sensitivity()
        self._generate_product()

    def _get_date(self) -> datetime:
        """Returns measurement date as datetime."""
        return datetime(
            int(self.dataset.year),
            int(self.dataset.month),
            int(self.dataset.day),
            0,
            0,
            0,
            tzinfo=timezone.utc,
        )

    def _get_radar_frequency(self) -> npt.NDArray | None:
        try:
            return self.getvar("radar_frequency")
        except (KeyError, RuntimeError):
            return None

    def _get_z_sensitivity(self) -> npt.NDArray | None:
        try:
            return self.getvar("Z_sensitivity")
        except (KeyError, RuntimeError):
            return None

    def _generate_product(self) -> None:
        """Process needed data of observation to a ObservationManager object."""
        try:
            if self.obs == "cf":
                self.append_data(self._generate_cf(), "cf")
            else:
                self.append_data(self.getvar(self.obs), self.obs)
                if self.obs == "iwc":
                    self._mask_iwc()
            self.append_data(self.getvar("height"), "height")
        except (KeyError, RuntimeError):
            msg = f"Failed to read {self.obs} from {self._file}"
            logging.exception(msg)
            raise

    def _generate_cf(self) -> npt.NDArray:
        """Generates cloud fractions using categorize bits and masking conditions."""
        categorize_bits = CategorizeBits(self._file)
        cloud_mask = self._classify_basic_mask(categorize_bits.category_bits)
        return self._mask_cloud_bits(cloud_mask)

    @staticmethod
    def _classify_basic_mask(bits: CategoryBits) -> npt.NDArray:
        cloud_mask = bits.droplet + bits.falling * 2
        cloud_mask[bits.falling & bits.freezing] = (
            cloud_mask[bits.falling & bits.freezing] + 2
        )
        cloud_mask[bits.aerosol] = 6
        cloud_mask[bits.insect] = 7
        cloud_mask[bits.aerosol & bits.insect] = 8
        return cloud_mask

    @staticmethod
    def _mask_cloud_bits(cloud_mask: npt.NDArray) -> npt.NDArray:
        """Creates cloud fraction."""
        for i in [1, 3, 4, 5]:
            cloud_mask[cloud_mask == i] = 1
        for i in [2, 6, 7, 8]:
            cloud_mask[cloud_mask == i] = 0
        return cloud_mask

    def _mask_iwc(self) -> None:
        """Keeps only reliable ice water content retrievals.

        Status 1 is a reliable retrieval and status 3 is a retrieval with the
        radar corrected for liquid, rain and melting attenuation; everything
        else (uncorrected attenuation, lidar-only, rain) is masked out.
        """
        iwc = self.getvar("iwc")
        iwc_status = self.getvar("iwc_retrieval_status")
        iwc[~np.isin(iwc_status, (1, 3))] = ma.masked
        self.append_data(iwc, "iwc")
