import os
from cloudnetpy.instruments import ceilo2nc, mira2nc
from cloudnetpy.categorize import generate_categorize
import netCDF4
import sys
import warnings
from cloudnetpy.quality import Quality

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(SCRIPT_PATH)
from all_products_fun import AllProductsFun
from radar_fun import RadarFun
from lidar_fun import LidarFun

site_meta = {
    'name': 'Munich',
    'altitude': 538,
    'latitude': 48.5,
    'longitude': 11.5
}
filepath = f'{SCRIPT_PATH}/../source_data/'
date = '2021-11-20'


class TestCategorize:

    radar_file = 'dummy_radar_file_for_cat.nc'
    lidar_file = 'dummy_lidar_file_for_cat.nc'

    uuid_radar = mira2nc(f'{filepath}raw_mira_radar.mmclx', radar_file, site_meta)
    uuid_lidar = ceilo2nc(f'{filepath}raw_chm15k_lidar.nc', lidar_file, site_meta)

    input_files = {
        'radar': radar_file,
        'lidar': lidar_file,
        'mwr': f'{filepath}hatpro_mwr.nc',
        'model': f'{filepath}ecmwf_model.nc'
    }

    output = 'dummy_categorize_file'
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=DeprecationWarning)
        uuid = generate_categorize(input_files, output)
    nc = netCDF4.Dataset(output)
    all_fun = AllProductsFun(nc, site_meta, date, uuid)
    radar_fun = RadarFun(nc, site_meta, date, uuid)
    lidar_fun = LidarFun(nc, site_meta, date, uuid)
    quality = Quality(output)
    res_data = quality.check_data()
    res_metadata = quality.check_metadata()

    def test_common(self):
        for name, method in AllProductsFun.__dict__.items():
            if 'test_' in name:
                getattr(self.all_fun, name)()

    def test_global_attributes(self):
        assert self.nc.title == 'Cloud categorization products from Munich'

    def test_qc(self):
        assert self.quality.n_metadata_test_failures == 0, self.res_metadata
        assert self.quality.n_data_test_failures == 0, self.res_data

    def test_tear_down(self):
        for file in (self.output, self.radar_file, self.lidar_file):
            os.remove(file)
        self.nc.close()
