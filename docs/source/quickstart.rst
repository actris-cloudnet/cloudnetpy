==========
Quickstart
==========

In this tutorial products are created using CloudnetPy's high level API.

Raw data conversion
-------------------

Cloudnet Level 1b netCDF are generated from raw data from instruments.
You can find raw data from the
`Cloudnet data portal API <https://docs.cloudnet.fmi.fi/api/data-portal.html#get-apiraw-files--upload>`_.
Alternatively, you can download `preprocessed example files <http://lake.fmi.fi/cloudnet-public/cloudnetpy_test_input_files.zip>`_
and jump to the next section on product generation.

Radar processing
~~~~~~~~~~~~~~~~

In the first example we convert a raw METEK MIRA-36 cloud radar file into
Cloudnet netCDF file that can be used in further processing steps.

.. code-block:: python

    from cloudnetpy.instruments import mira2nc
    uuid = mira2nc('raw_mira_radar.mmclx', 'radar.nc', {'name': 'Mace-Head'})

where ``uuid`` is an unique identifier for the generated ``radar.nc`` file.
For more information, see `API reference <api.html#instruments.mira2nc>`__ for this function.

Lidar processing
~~~~~~~~~~~~~~~~

Next we convert a raw Lufft CHM 15k ceilometer (lidar) file into Cloudnet netCDF file
and process the signal-to-noise screened backscatter coefficient. Also this converted lidar
file will be needed later.

.. code-block:: python

    from cloudnetpy.instruments import ceilo2nc
    uuid = ceilo2nc('raw_chm15k_lidar.nc', 'lidar.nc', {'name':'Mace-Head', 'altitude':5})

where ``uuid`` is an unique identifier for the generated ``lidar.nc`` file.
For more information, see `API reference <api.html#instruments.ceilo2nc>`__ for this function.

MWR processing
~~~~~~~~~~~~~~

Next we convert RPG-HATPRO microwave radiometer (MWR) binary files (e.g. \*.LWP) into Cloudnet
netCDF file to retrieve integrated liquid water path (LWP).

.. code-block:: python

    from cloudnetpy.instruments import hatpro2nc
    uuid = hatpro2nc('path/to/hatpro-raw-files/', 'hatpro_mwr.nc', {'name':'Mace-Head', 'altitude':5})

where ``uuid`` is an unique identifier for the generated ``hatpro_mwr.nc`` file.
For more information, see `API reference <api.html#instruments.hatpro2nc>`__ for this function.

However, with a 94 GHz RPG cloud radar, a separate MWR instrument is not necessarily
required. RPG radars contain single MWR channel providing LWP measurements, which can be
used in CloudnetPy. Nevertheless, it is always recommended to equip a measurement site
with a dedicated multi-channel radiometer if possible.

Model data
~~~~~~~~~~

Model files needed in the next processing step can be downloaded
from the `Cloudnet data portal API <https://docs.cloudnet.fmi.fi/api/data-portal.html#get-apimodel-files--modelfile>`_.
Several models may be available depending on the site and date.
The list of different model models can be found `here <https://cloudnet.fmi.fi/api/models/>`_.

Product generation
------------------

After processing the raw radar, lidar and MWR files, and acquiring
a model file, Cloudnet products can be created.

Categorize processing
~~~~~~~~~~~~~~~~~~~~~

In the next example we create a categorize file starting from the
``radar.nc``, ``hatpro_mwr.nc`` and ``lidar.nc`` files generated above. The required
``ecmwf_model.nc`` file is
included in the provided `example files <http://lake.fmi.fi/cloudnet-public/cloudnetpy_test_input_files.zip>`_.

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


Classification processing
~~~~~~~~~~~~~~~~~~~~~~~~~

In the last example we create the smallest and simplest Cloudnet
product, the classification product. The product-generating functions always
use a categorize file as an input.

.. code-block:: python

    from cloudnetpy.products import generate_classification
    uuid = generate_classification('categorize.nc', 'classification.nc')

where ``uuid`` is an unique identifier for the generated ``classification.nc`` file.
Corresponding functions are available for other products
(see :ref:`Product generation`).
