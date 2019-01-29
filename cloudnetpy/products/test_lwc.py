#!venv/bin/python3
import lwc
import ncf

# test file
#fname = '20170927_ny-alesund_categorize.nc'
#fname = '20170608_lindenberg_categorize.nc'
fname = 'data/20171010_ny-alesund_categorize.nc'

# generate liquid water content
for n in range(10):
    print(n)
    (cat, obs) = lwc.generate_lwc(fname)
    ncf.save_Cnet(cat, obs, 'test_lwc.nc', 'Liquid water content', 0.1)



