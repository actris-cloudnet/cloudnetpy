"""Module for reading raw cloud radar data."""
import os
from typing import Union, List, Optional
from tempfile import NamedTemporaryFile
import numpy as np
from cloudnetpy import output, utils
from cloudnetpy.instruments.nc_radar import NcRadar
from cloudnetpy.metadata import MetaData
from cloudnetpy import concat_lib


def mira2nc(raw_mira: str,
            output_file: str,
            site_meta: dict,
            rebin_data: Optional[bool] = False,
            keep_uuid: Optional[bool] = False,
            uuid: Optional[str] = None,
            date: Optional[str] = None) -> str:
    """Converts METEK MIRA-35 cloud radar data into Cloudnet Level 1b netCDF file.

    This function converts raw MIRA file(s) into a much smaller file that
    contains only the relevant data and can be used in further processing
    steps.

    Args:
        raw_mira: Filename of a daily MIRA .mmclx file. Can be also a folder containing several
            non-concatenated .mmclx files from one day.
        output_file: Output filename.
        site_meta: Dictionary containing information about the site. Required key value pair
            is `name`.
        rebin_data: If True, rebins data to 30s resolution. Otherwise keeps the native resolution.
            Default is False.
        keep_uuid: If True, keeps the UUID of the old file, if that exists. Default is False when
            new UUID is generated.
        uuid: Set specific UUID for the file.
        date: Expected date as YYYY-MM-DD of all profiles in the file.

    Returns:
        UUID of the generated file.

    Raises:
        ValueError: Timestamps from several days or timestamps do not match the expected date.

    Examples:
          >>> from cloudnetpy.instruments import mira2nc
          >>> site_meta = {'name': 'Vehmasmaki'}
          >>> mira2nc('raw_radar.mmclx', 'radar.nc', site_meta)
          >>> mira2nc('/one/day/of/mira/mmclx/files/', 'radar.nc', site_meta)

    """
    keymap = {'Zg': 'Ze',
              'VELg': 'v',
              'RMSg': 'width',
              'LDRg': 'ldr',
              'SNRg': 'SNR'}

    if os.path.isdir(raw_mira):
        temp_file = NamedTemporaryFile()
        mmclx_filename = temp_file.name
        valid_filenames = utils.get_sorted_filenames(raw_mira, '.mmclx')
        concat_lib.concatenate_files(valid_filenames, mmclx_filename, variables=list(keymap.keys()))
    else:
        mmclx_filename = raw_mira

    mira = Mira(mmclx_filename, site_meta)
    mira.init_data(keymap)
    mira.validate_date(date)
    mira.linear_to_db(('Ze', 'ldr', 'SNR'))
    if rebin_data:
        snr_gain = mira.rebin_fields()
    else:
        snr_gain = 1
    mira.screen_by_snr(snr_gain)
    mira.add_meta()
    mira.add_geolocation()
    mira.close()
    attributes = output.add_time_attribute(ATTRIBUTES, mira.date)
    output.update_attributes(mira.data, attributes)
    fields_from_source = ('nfft', 'prf', 'nave', 'zrg', 'rg0', 'drg')
    return output.save_radar_level1b(mmclx_filename, mira, output_file, keep_uuid, uuid,
                                     fields_from_source)


class Mira(NcRadar):
    """Class for MIRA-35 raw radar data. Child of NcRadar().

    Args:
        full_path (str): Filename of a daily MIRA .mmclx NetCDF file.
        site_meta (dict): Site properties in a dictionary. Required
            keys are: `name`.

    """
    radar_frequency = 35.5
    epoch = (1970, 1, 1)

    def __init__(self, full_path: str, site_meta: dict):
        super().__init__(full_path, site_meta)
        self.date = self._init_mira_date()
        self.source = 'METEK MIRA-35'

    def validate_date(self, expected_date: Union[str, None]) -> None:
        """Validates MIRA date timestamps."""
        time_stamps = self.getvar('time')
        for t in time_stamps[1:-1]:  # Ignore last timestamp (it can be 24:00)
            date = utils.seconds2date(t, self.epoch)[:3]
            if date != self.date:
                raise ValueError('Error: MIRA dates from different days.')
            if expected_date is not None and '-'.join(date) != expected_date:
                raise ValueError('Error: MIRA date differs from expected.')

    def add_geolocation(self):
        """Adds geo info (from global attributes to variables)."""
        for key in ('Latitude', 'Longitude', 'Altitude'):
            value = getattr(self.dataset, key).split()[0]
            key = key.lower()
            if key not in self.data.keys():  # Not provided by user
                self.append_data(value, key)

    def screen_by_snr(self, snr_gain: Optional[int] = 1, snr_limit: Optional[int] = -17) -> None:
        """Screens by SNR."""
        ind = np.where(self.data['SNR'][:] * snr_gain < snr_limit)
        for field in self.data:
            if self.data[field].data.ndim == 2:  # Screen only 2d variables
                self.data[field].mask_indices(ind)

    def _init_mira_date(self) -> List[str]:
        time_stamps = self.getvar('time')
        return utils.seconds2date(time_stamps[0], self.epoch)[:3]

    def rebin_fields(self):
        """Rebins fields."""
        time_grid = utils.time_grid()
        for field in self.data:
            self.data[field].rebin_data(self.time, time_grid)
        snr_gain = self._estimate_snr_gain(time_grid, self.time)
        self.time = time_grid
        return snr_gain

    @staticmethod
    def _estimate_snr_gain(time_sparse, time_dense):
        """Returns factor for SNR (dB) increase when data is binned."""
        binning_ratio = utils.mdiff(time_sparse) / utils.mdiff(time_dense)
        return np.sqrt(binning_ratio)


ATTRIBUTES = {
    'Ze': MetaData(
        long_name='Radar reflectivity factor (uncorrected), vertical polarization',
        units='dBZ',
    ),
    'SNR': MetaData(
        long_name='Signal-to-noise ratio',
        units='dB',
    )
}
