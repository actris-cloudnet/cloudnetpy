==========
Quickstart
==========

Processing is easy using CloudnetPy's high level APIs.
First, download some raw data from the
`Cloudnet data portal API <https://docs.cloudnet.fmi.fi/api/data-portal.html#get-apiraw-files--upload>`_
or try these `example files <http://lake.fmi.fi/cloudnet-public/cloudnetpy_test_input_files.zip>`_.

Radar processing
----------------

In the first example we convert a raw METEK MIRA-36 cloud radar file into
Cloudnet netCDF file that can be used in further processing steps.

.. code-block:: python

    from cloudnetpy.instruments import mira2nc
    uuid = mira2nc('raw_mira_radar.mmclx', 'radar.nc', {'name': 'Mace-Head'})

where ``uuid`` is an unique identifier for the generated ``radar.nc`` file.
For more information, see `API reference <api.html#instruments.mira2nc>`__ for this function.

Lidar processing
----------------

Next we convert a raw Lufft CHM15k ceilometer (lidar) file into Cloudnet netCDF file
and process the signal-to-noise screened backscatter coefficient. Also this converted lidar
file will be needed later.

.. code-block:: python

    from cloudnetpy.instruments import ceilo2nc
    uuid = ceilo2nc('raw_chm15k_lidar.nc', 'lidar.nc', {'name':'Mace-Head', 'altitude':5})

where ``uuid`` is an unique identifier for the generated ``lidar.nc`` file.
For more information, see `API reference <api.html#instruments.ceilo2nc>`__ for this function.

MWR processing
--------------

Processing of multi-channel HATPRO microwave radiometer (MWR) data is not part of CloudnetPy.
Thus, site operators need to run third-party processing software to retrieve integrated liquid
water path (LWP) from raw HATPRO brightness temperature measurements.

However, with a 94 GHz RPG cloud radar, a separate MWR instrument is not necessarily
required. RPG radars contain single MWR channel providing LWP measurements, which can be
used in CloudnetPy. Nevertheless, it is always recommended to equip a measurement site
with a dedicated multi-channel radiometer if possible.

Model data
----------

Model files needed in the next processing step can be downloaded
from the `Cloudnet http API <https://actris-cloudnet.github.io/dataportal/>`_.
Several models may be available depending on the site and date.
The list of different model models can be found `here <https://cloudnet.fmi.fi/api/models/>`_.

Categorize processing
---------------------

After processing the raw radar and raw lidar files, and acquiring
the model and mwr files, a Cloudnet categorize file can be created.

In the next example we create a categorize file starting from the
``radar.nc`` and ``lidar.nc`` files generated above. The required
``ecmwf_model.nc`` and ``hatpro_mwr.nc`` files are
included in the provided `example input files <http://devcloudnet.fmi.fi/files/cloudnetpy_test_input_files.zip>`_.

.. code-block:: python

   from cloudnetpy.categorize import generate_categorize
   input_files = {
       'radar': 'radar.nc',
       'lidar': 'lidar.nc',
       'model': 'ecmwf_model.nc',
       'mwr': 'hatpro_mwr.nc'
       }
   uuid = generate_categorize(input_files, 'categorize.nc')

where ``uuid`` is an unique identifier for the generated ``categorize.nc`` file.
For more information, see `API reference <api.html#categorize.generate_categorize>`__ for this function.
Note that with a 94 GHz RPG cloud radar, the ``radar.nc`` file can be used as input
for both inputs: ``'radar'`` and ``'mwr'``.


Processing products
-------------------

In the last example we create the smallest and simplest Cloudnet
product, the classification product. The product-generating functions always
use a categorize file as an input.

.. code-block:: python

    from cloudnetpy.products import generate_classification
    uuid = generate_classification('categorize.nc', 'classification.nc')

where ``uuid`` is an unique identifier for the generated ``classification.nc`` file.
Corresponding functions are available for other products
(see :ref:`Product generation`).
