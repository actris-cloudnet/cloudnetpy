High level APIs
===============

CloudnetPy's high level APIs provide a simple way to process
cloud remote sensing measurements into Cloudnet products, without
needing to worry about implementation details too much.

High level APIs enable users to set up an operational processing
system that doesn't require major modifications
if the underlaying methods and their implementations change
from one software version to another.

.. autofunction:: categorize.generate_categorize



Modules
=======

The various modules of CloudnetPy also provide additional lower level functions
that can be useful for development, testing and research purposes.

cloudnet.categorize
-------------------

.. automodule:: categorize
   :members:
   :exclude-members: generate_categorize


cloudnet.classify
-----------------

.. automodule:: classify
   :members:


cloudnet.atmos
--------------

.. automodule:: atmos
   :members:


cloudnet.droplet
----------------

.. automodule:: droplet
   :members:


cloudnet.lwc
------------

.. automodule:: lwc
   :members:


cloudnet.ncf
------------

.. automodule:: ncf
   :members:

      
cloudnet.utils
--------------

.. automodule:: utils
   :members:





