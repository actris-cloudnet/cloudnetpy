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


.. autofunction:: instruments.mira.mira2nc

.. autofunction:: instruments.rpg.rpg2nc

.. autofunction:: instruments.ceilo.ceilo2nc

.. autofunction:: categorize.categorize.generate_categorize


Categorize to products
......................

Starting from the categorize file, several geophysical products can be
generated.

.. autofunction:: products.classification.generate_classification

.. autofunction:: products.iwc.generate_iwc

.. autofunction:: products.lwc.generate_lwc

.. autofunction:: products.drizzle.generate_drizzle


Visualizing results
...................

CloudnetPy offers easy-to-use plotting interface.

.. autofunction:: plotting.plotting.generate_figure

There is also possibility to compare old Cloudnet files with the new
CloudnetPy files.

.. autofunction:: plotting.plotting.compare_files


Categorize subpackage
---------------------

Categorize is a CloudnetPy's subpackage. It contains
several modules that are used when creating the Cloudnet
categorize file.


cloudnetpy.categorize.classify
..............................

.. automodule:: categorize.classify
   :members:


cloudnetpy.categorize.atmos
...........................

.. automodule:: categorize.atmos
   :members:


cloudnetpy.categorize.droplet
.............................

.. automodule:: categorize.droplet
   :members:


Products subpackage
-------------------

Products is a CloudnetPy's subpackage. It contains
several modules that correspond to different Cloudnet
products.

cloudnetpy.products.classification
..................................

.. automodule:: products.classification
   :members:


cloudnetpy.products.iwc
.......................

.. automodule:: products.iwc
   :members:


cloudnetpy.products.lwc
.......................

.. automodule:: products.lwc
   :members:


cloudnetpy.products.drizzle
...........................

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


