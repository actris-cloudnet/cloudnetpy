#!venv/bin/python3
from iwc import generate_iwc
from ncf import save_Cnet

# test file
fname = 'data/20170927_ny-alesund_categorize.nc'
#fname = '20170608_lindenberg_categorize.nc'
#fname = '20171010_ny-alesund_categorize.nc'

# generate ice water content
for n in range(10):
    print(n)
    (cat, obs) = generate_iwc(fname)
    save_Cnet(cat, obs, 'test_iwc.nc', 'Ice water content', 0.1)
    

