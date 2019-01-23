""" Main testing file for Categorize file creation 

Test data is here:

http://tukiains.kapsi.fi/omia/test_data.tar.gz

"""

import sys
sys.path.insert(0, '../cloudnetpy')
import cloudnetpy.categorize as cat


def main():
    """ Main function. """

    # Input must be a tuple containing full paths of 
    # the 4 required files: (radar, lidar, mwr, model),
    # given in this order!
    input_files = (
        '../test_data/20160614_mace-head_mira.nc',
        #'../test_data/radar_test_file.nc',
        '../test_data/20160614_mace-head_chm15k.nc',
        '../test_data/160614.LWP.NC',
        '../test_data/20160614_mace-head_gdas1.nc')

    # Output file name (and path, optionally).
    output_file = '../test_data/categorize_test_file.nc'

    cat.generate_categorize(input_files, output_file)


if __name__ == "__main__":
    main()
