""" Main testing file for Categorize file creation 

Test data is here:

http://tukiains.kapsi.fi/omia/test_data.tar.gz

"""

import sys
sys.path.insert(0, '../cloudnetpy')
import cloudnetpy.categorize as cat
from cloudnetpy import plotting as plot

def main():
    """ Main function. """

    input_files = {
        'radar': '../test_data/20180611_hyytiala_rpg.nc',
        'lidar': '../test_data/20160614_mace-head_chm15k.nc',
        'model': '../test_data/20180611_hyytiala_icon-iglo-12-23.nc',
        'mwr': '../test_data/20180611_hyytiala_rpg.nc',
        }

    input_files2 = {
        'radar': '../test_data/20160614_mace-head_mira.nc',
        'lidar': '../test_data/20160614_mace-head_chm15k.nc',
        'model': '../test_data/20160614_mace-head_gdas1.nc',
        'mwr': '../test_data/160614.LWP.NC'
        }


    # Output file name (and path, optionally).
    output_file = '../test_data/categorize_test_file.nc'

    cat.generate_categorize(input_files, output_file)

    f2 = '/home/tukiains/Documents/PYTHON/cloudnetpy/test_data/20160614_mace-head_categorize.nc'

    #plot.plot_overview(output_file, '20160614', ylim=(0, 360))
    plot.plot_variable(output_file, f2, 'melting', '20160614', ylim=(0, 360))


if __name__ == "__main__":
    main()
