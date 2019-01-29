from classification_vanha import generate_class
from ncf import save_Cnet

# test file
#fname = '20170608_lindenberg_categorize.nc'
#fname = '20170927_ny-alesund_categorize.nc'
fname = '/home/korpinen/Documents/ACTRIS/cloudnet_data/20190113_juelich_categorize.nc'
#fname = 'data/20180110_mace-head_categorize.nc'

# generate classification
for n in range(1):
    print(n)
    (cat, obs) = generate_class(fname)
    save_Cnet(cat, obs, 'test_class.nc', 'Classification', 0.1)
    

