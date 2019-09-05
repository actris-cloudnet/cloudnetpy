==========
Quickstart
==========

Processing is easy using CloudnetPy's high level APIs.

Radar processing
----------------

In the first example we convert a raw METEK MIRA-36 netCDF file into
Cloudnet netCDF file that can be used in further processing steps.

.. code-block:: python

    from cloudnetpy.instruments.mira import mira2nc
    mira2nc('raw_radar.mmclx', 'radar.nc', {'name': 'Mace-Head'})

Lidar processing
----------------

Next we convert a raw Vaisala ceilometer text file into netCDF (and process
the signal-to-noise screened backscatter).

.. code-block:: python

    from cloudnetpy.instruments.ceilo import ceilo2nc
    ceilo2nc('vaisala.txt', 'vaisala.nc', {'name':'Kumpula', 'altitude':53})

The same function can handle also Jenoptik CHM15k files.

.. code-block:: python

    ceilo2nc('jenoptik_chm15k.nc', 'jenoptik.nc', {'name':'Mace Head', 'altitude':16})


MWR processing
--------------

Processing of multi-channel HATPRO microwave radiometer (MWR) data is not yet part of CloudnetPy.
Thus, site operators need to run custom processing software to retrieve integrated liquid
water path (LWP) from raw HATPRO measurements.

However, with a 94 GHz RPG cloud radar, a separate MWR instrument is not necessarely
required. RPG radars contain single MWR channel providing a rough estimate
of LWP, which can be used in CloudnetPy. Nevertheless, it is always
recommended to equip measurement site with a dedicated multi-channel
radiometer if possible.

Model data
----------

Model files needed in the next processing step can be downloaded
from `Cloudnet http API <http://devcloudnet.fmi.fi/api/>`_. Several models
may be available depending on the site and date, see for example
`this day <http://devcloudnet.fmi.fi/api/models/?site_code=mace-head&date=20190303>`_.
Any model file can be used in the processing but the recommended order is

#. ecmwf
#. icon-iglo-12-23
#. gdas1

Categorize processing
---------------------

In the next example we create a categorize file from already
calibrated measurement files.

.. code-block:: python

   from cloudnetpy.categorize.categorize import generate_categorize
   input_files = {
       'radar': 'radar.nc',
       'lidar': 'lidar.nc',
       'model': 'model.nc',
       'mwr': 'mwr.nc'
       }
   generate_categorize(input_files, 'categorize.nc')

With a 94 GHz RPG cloud radar, the radar.nc file can be used for both 'radar' and 'mwr'.


Processing products
-------------------

In the last example we create the smallest and simplest Cloudnet
product, the classification product.

.. code-block:: python

    from cloudnetpy.products.classification import generate_classification
    generate_classification('categorize.nc', 'classification.nc')

