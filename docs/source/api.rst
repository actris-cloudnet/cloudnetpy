API reference
=============


High-level functions
--------------------

CloudnetPy's high-level functions provide a simple mechanism to process
cloud remote sensing measurements into Cloudnet products. A full processing
goes in steps. Each step produces a file which used as an input for the
next step.

Raw data to categorize
......................

Cloudnet instruments provide raw data in various formats that need to
be first converted into netCDF with standardized metadata. After that,
the data can be combined in a single categorize file.


.. autofunction:: instruments.mira2nc

.. autofunction:: instruments.rpg2nc

.. autofunction:: instruments.ceilo2nc

.. autofunction:: categorize.generate_categorize


Categorize to products
......................

Starting from the categorize file, several geophysical products can be
generated.

.. autofunction:: products.generate_classification

.. autofunction:: products.generate_iwc

.. autofunction:: products.generate_lwc

.. autofunction:: products.generate_drizzle


Visualizing results
...................

CloudnetPy offers an easy-to-use plotting interface:

.. autofunction:: plotting.generate_figure

There is also possibility to compare CloundetPy files with the
Matlab-processed old files which are available from `devcloudnet.fmi.fi
<http://devcloudnet.fmi.fi>`_:


.. autofunction:: plotting.compare_files


Quality control
...............

CloudnetPy Github source contains functions to check the quality of the
processed files. These can be used to validate the metadata of the files
and perform various checks for the data.

.. autofunction:: tests.check_metadata

.. autofunction:: tests.check_data_quality

.. autofunction:: tests.run_unit_tests

.. note::

    Quality control routines are not included in the CloudnetPy PyPI
    package.


Categorize modules
------------------

Categorize is a CloudnetPy's subpackage. It contains
several modules that are used when creating the Cloudnet
categorize file.


datasource
..........

.. automodule:: categorize.datasource
   :members:

radar
.....

.. automodule:: categorize.radar
   :members:

lidar
.....

.. automodule:: categorize.lidar
   :members:

mwr
...

.. automodule:: categorize.mwr
   :members:

model
.....

.. automodule:: categorize.model
   :members:

classify
........

.. automodule:: categorize.classify
   :members:

melting
.......

.. automodule:: categorize.melting
   :members:

freezing
........

.. automodule:: categorize.freezing
   :members:


falling
.......

.. automodule:: categorize.falling
   :members:


insects
.......

.. automodule:: categorize.insects
   :members:


atmos
.....

.. automodule:: categorize.atmos
   :members:


droplet
.......

.. automodule:: categorize.droplet
   :members:


Products modules
----------------

Products is a CloudnetPy's subpackage. It contains
several modules that correspond to different Cloudnet
products.

classification
..............

.. automodule:: products.classification
   :members:


iwc
...

.. automodule:: products.iwc
   :members:


lwc
...

.. automodule:: products.lwc
   :members:


drizzle
.......

.. automodule:: products.drizzle
   :members:


Misc
----

Documentation for various modules with low-level
functionality.

cloudnetpy.utils
................

.. automodule:: utils
   :members:


