""" Main testing file for Categorize file creation """
import sys
import os
sys.path.insert(0, '../cloudnetpy')

import cloudnetpy.categorize as cat

def main():
    """ Main function. """

    radar_file = 'test_data/20180110_mace-head_mira.nc'
    lidar_file = 'test_data/20180110_mace-head_chm15k.nc'
    mwr_file = 'test_data/180110.LWP.NC'
    model_file = 'test_data/20180110_mace-head_gdas1.nc'
    input_files = (radar_file, lidar_file, mwr_file, model_file)
    output_file = 'test_cat.nc'
    site = 'Mace Head'
    institute = 'Finnish Meteorological Institute'
    aux = (site, institute)

    cat.generate_categorize(input_files, output_file, aux)


if __name__ == "__main__":
    main()
