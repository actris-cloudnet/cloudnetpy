High-level APIs
===============

CloudnetPy's high-level APIs provide a simple way to process
cloud remote sensing measurements into Cloudnet products, without
needing to worry about implementation details.

The APIs enable users to set up an operational processing
system that doesn't require major modifications
if the underlaying methods and their implementations change
from one software version to another.

.. autofunction:: mira.mira2nc

.. autofunction:: rpg.rpg2nc

.. autofunction:: categorize.generate_categorize



Modules
=======

The various modules of CloudnetPy provide additional lower-level
functions that are useful for development, testing and research
purposes.

cloudnetpy.categorize
---------------------

.. automodule:: categorize
   :members:
   :exclude-members: generate_categorize
   :member-order: bysource


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


cloudnetpy.cloudnetarray
------------------------

.. automodule:: cloudnetarray
   :members:


cloudnetpy.utils
----------------

.. automodule:: utils
   :members:





