import sys
sys.path.insert(0,'/home/korpinen/Documents/ACTRIS/cloudnetpy/')
from cloudnetpy.products.classification import generate_class

# test file
fname = '/home/korpinen/Documents/ACTRIS/cloudnet_data/20190113_juelich_categorize.nc'
outname = '/home/korpinen/Documents/ACTRIS/cloudnetpy/test_data.nc'

# generate classification
generate_class(fname,outname)