API reference
=============


High-level functions
--------------------

CloudnetPy's high-level functions provide a simple mechanism to process
cloud remote sensing measurements into Cloudnet products. A full processing
goes in steps. Each step produces a file which used as an input for the
next step.

Raw data conversion
...................

Different Cloudnet instruments provide raw data in various formats (netCDF, binary, text)
that first need to be converted into homogeneous Cloudnet netCDF files
containing harmonized units and other metadata. This initial processing step
is necessary to ensure that the subsequent processing steps work with
all supported instrument combinations.

.. autofunction:: instruments.mira2nc

.. autofunction:: instruments.rpg2nc

.. autofunction:: instruments.ceilo2nc


The categorize file
...................

The categorize file concatenates all input data into common
time / height grid.

.. autofunction:: categorize.generate_categorize


Product generation
..................

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

Categorize is CloudnetPy's subpackage. It contains
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

Products is CloudnetPy's subpackage. It contains
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


product_tools
.............

.. automodule:: products.product_tools
   :members:


Misc
----

Documentation for various modules with low-level
functionality.


utils
.....

.. automodule:: utils
   :members:


cloudnetarray
.............

.. automodule:: cloudnetarray
   :members:


output
......

.. automodule:: output
   :members:

