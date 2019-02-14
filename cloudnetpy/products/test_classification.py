import sys
sys.path.insert(0,'/home/korpinen/Documents/ACTRIS/cloudnetpy/')
from cloudnetpy.products.classification import generate_class
from cloudnetpy.products.ncf import save_Cnet

# test file
#fname = '20170608_lindenberg_categorize.nc'
#fname = '20170927_ny-alesund_categorize.nc'
fname = '/home/korpinen/Documents/ACTRIS/cloudnet_data/20190113_juelich_categorize.nc'
outname = '/home/korpinen/Documents/ACTRIS/cloudnetpy/test_data.nc'
#fname = 'data/20180110_mace-head_categorize.nc'

# generate classification
generate_class(fname,outname)