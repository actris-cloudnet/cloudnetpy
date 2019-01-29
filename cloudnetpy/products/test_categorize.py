#!venv/bin/pytho3
import categorize as cat
import ncf

# test input files:
#model_file = 'data/20180110_mace-head_gdas1.nc'
#radar_file = 'data/20180110_mace-head_mira.nc'
#lidar_file = 'data/20180110_mace-head_chm15k.nc'
mwr_file   = 'data/180110.LWP.NC'

# 8.8.2017 mace head
#model_file = 'data/20170808_mace-head_gdas1.nc'
#radar_file = 'data/20170808_mace-head_mira.nc'
#lidar_file = 'data/20170808_mace-head_chm15k.nc'

# 20.6.2013 Sodankyla
model_file = 'data/20130620_sodankyla_gdas1.nc'
radar_file = 'data/20130620_sodankyla_mira.nc'
lidar_file = 'data/20130620_sodankyla_ct25k.nc'

# output file
output_file = 'test_cat.nc' 

# auxillary information 
site        = 'Mace Head'
institute   = 'Finnish Meteorological Institute'

aux = (site, institute)
       
# generate categorize file
cat.generate_categorize(radar_file, lidar_file, model_file, mwr_file, output_file, aux)

