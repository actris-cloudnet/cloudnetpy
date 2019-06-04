""" Main testing file for Categorize file creation 

"""
import sys
import cloudnetpy.categorize as cat
from cloudnetpy.instruments import mira, ceilo
sys.path.insert(0, '../cloudnetpy')


def main():
    """ Main function. """

    prefix = '/home/korpinen/Documents/ACTRIS/cloudnet_data/'
    mira_raw = prefix + '20181204_mace-head_mira_raw.nc'
    mira.mira2nc(mira_raw, prefix + 'mira_test.nc', {'name': 'Mace-Head'})
    input_files = {
        'radar': prefix + 'mira_test.nc',
        'lidar': prefix + '20181204_mace-head_chm15k.nc',
        'model': prefix + '20181204_mace-head_ecmwf.nc',
        'mwr': prefix + '20181204_mace-head_hatpro.nc',
        }

    output_file = prefix + 'categorize_test_file_new.nc'

    cat.generate_categorize(input_files, output_file)


if __name__ == "__main__":
    main()
