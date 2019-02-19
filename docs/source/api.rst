High-level APIs
===============

CloudnetPy's high-level APIs provide a simple way to process
cloud remote sensing measurements into Cloudnet products, without
needing to worry about implementation details.

The APIs enable users to set up an operational processing
system that doesn't require major modifications
if the underlaying methods and their implementations change
from one software version to another.

Raw data to categorize
----------------------

Cloudnet instruments provide raw data in various formats that need to
be first converted into netCDF with standardized metadata. After that,
the data can be combined in a single categorize file.


.. autofunction:: mira.mira2nc

.. autofunction:: rpg.rpg2nc

.. autofunction:: categorize.generate_categorize


Categorize to products
----------------------

Starting from the categorize file, several geophysical products can be
generated.

.. autofunction:: products.classification.generate_class


Cloudnetpy modules
==================

The various modules of CloudnetPy provide additional lower-level
functions that are useful for development, testing and research
purposes.


cloudnetpy.classify
-------------------

.. automodule:: classify
   :members:


cloudnetpy.atmos
----------------

.. automodule:: atmos
   :members:


cloudnetpy.droplet
------------------

.. automodule:: droplet
   :members:


cloudnetpy.lwc
--------------

.. automodule:: lwc
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


