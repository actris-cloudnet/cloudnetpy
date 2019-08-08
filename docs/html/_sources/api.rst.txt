High-level APIs
===============

CloudnetPy's high-level APIs provide a simple way to process
cloud remote sensing measurements into Cloudnet products, without
needing to worry about implementation details.

The APIs enable users to set up an operational processing
system that doesn't require major modifications
if the underlying methods and their implementations change
from one software version to another.

Raw data to categorize
----------------------

Cloudnet instruments provide raw data in various formats that need to
be first converted into netCDF with standardized metadata. After that,
the data can be combined in a single categorize file.


.. autofunction:: instruments.mira.mira2nc

.. autofunction:: instruments.rpg.rpg2nc

.. autofunction:: instruments.ceilo.ceilo2nc

.. autofunction:: categorize.categorize.generate_categorize


Categorize to products
----------------------

Starting from the categorize file, several geophysical products can be
generated.

.. autofunction:: products.classification.generate_classification

.. autofunction:: products.iwc.generate_iwc

.. autofunction:: products.lwc.generate_lwc

.. autofunction:: products.drizzle.generate_drizzle


Visualizing results
-------------------

CloudnetPy offers easy-to-use plotting interface.

.. autofunction:: plotting.plotting.generate_figure

There is also possibility to compare old Cloudnet files with the new
CloudnetPy files.

.. autofunction:: plotting.plotting.compare_files


Cloudnetpy modules
==================

The various modules of CloudnetPy provide additional lower-level
functions that are useful for development, testing and research
purposes.


cloudnetpy.categorize.classify
------------------------------

.. automodule:: categorize.classify
   :members:


cloudnetpy.categorize.atmos
---------------------------

.. automodule:: categorize.atmos
   :members:


cloudnetpy.categorize.droplet
-----------------------------

.. automodule:: categorize.droplet
   :members:


cloudnetpy.utils
----------------

.. automodule:: utils
   :members:


Products modules
================

Products is a CloudnetPy's sub-package. It contains
several modules that correspond to different Cloudnet
products.

products.classification
-----------------------

.. automodule:: products.classification
   :members:


products.iwc
-----------------------

.. automodule:: products.iwc
   :members:


products.lwc
-----------------------

.. automodule:: products.lwc
   :members:


products.drizzle
-----------------------

.. automodule:: products.drizzle
   :members:


