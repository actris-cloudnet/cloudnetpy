=========================
Installation Instructions
=========================

CloudnetPy can be installed on any computer supporting Python3.6 (or higher).
The actual installation procedure depends on the operating system. The
instructions below are for Ubuntu.

Python Installation
-------------------

.. code-block:: console
		
   $ sudo apt update && sudo apt upgrade
   $ sudo apt install python3 python3-venv python3-pip python3-tk

Virtual Environment
-------------------

Create a new virtual environment and activate it:

.. code-block:: console
		
   $ python3 -m venv venv
   $ source venv/bin/activate


Pip-based Installation
----------------------

CloudnetPy is available from Python Package Index, `PyPI
<https://pypi.org/project/cloudnetpy/>`_.
Use Python's package manager, `pip <https://pypi.org/project/pip/>`_,
to install CloudnetPy package into the virtual environment:

.. code-block:: console
		
   (venv)$ pip3 install cloudnetpy

CloudnetPy is now ready for use from that virtual environment.

.. note::

   CloudnetPy codebase is rapidly developing and the PyPI package does not
   necessarily contain all the latest features and modifications. To get an up-to-date
   version of CloudnetPy, download it directly from `GitHub
   <https://github.com/actris-cloudnet/cloudnetpy>`_.


