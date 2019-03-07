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

    prefix = '/home/tukiains/Documents/PYTHON/cloudnetpy/test_data/'

    input_files = {
        'radar': prefix + '20181204_mace-head_mira.nc',
        'lidar': prefix + '20181204_mace-head_chm15k.nc',
        'model': prefix + '20181204_mace-head_ecmwf.nc',
        'mwr': prefix + '20181204_mace-head_hatpro.nc',
        }

    output_file = prefix + 'categorize_test_file.nc'

    cat.generate_categorize(input_files, output_file)

    #import netCDF4
    #category_bits = netCDF4.Dataset(output_file).variables['category_bits'][:]
    #plot.plot_2d(category_bits, cmap='Set1', ncolors=6)

    #f2 = '/home/tukiains/Documents/PYTHON/cloudnetpy/test_data/20160614_mace-head_categorize.nc'
    #plot.plot_overview(output_file, '20160614', ylim=(0, 360), savefig=True, savepath='/home/tukiains/Pictures/')
    #plot.plot_variable(output_file, f2, 'melting', '20160614', ylim=(0, 360))


if __name__ == "__main__":
    main()
