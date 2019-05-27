""" Main testing file for Categorize file creation 

"""
import sys
import cloudnetpy.categorize as cat
sys.path.insert(0, '../cloudnetpy')


def main():
    """ Main function. """

    prefix = '/home/korpinen/Documents/ACTRIS/cloudnet_data/'

    input_files = {
        'radar': prefix + '20181204_mace-head_mira.nc',
        'lidar': prefix + '20181204_mace-head_chm15k.nc',
        'model': prefix + '20181204_mace-head_ecmwf.nc',
        'mwr': prefix + '20181204_mace-head_hatpro.nc',
        }

    output_file = prefix + 'categorize_test_file_new.nc'

    cat.generate_categorize(input_files, output_file)


if __name__ == "__main__":
    main()
