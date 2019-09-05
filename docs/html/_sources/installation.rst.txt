=========================
Installation Instructions
=========================

CloudnetPy can be installed on any computer supporting Python.
The actual installation procedure depends on the operating system. The
instructions below are for Ubuntu, although they work with
many other Linux systems as well.

Python Installation
-------------------

CloudnetPy requires Python3.7 or higher:

.. code-block:: console
		
   $ sudo apt update && sudo apt upgrade
   $ sudo apt install python3.7 python3.7-venv python3-pip

Virtual Environment
-------------------

Create a new virtual environment and activate it:

.. code-block:: console
		
   $ python3.7 -m venv venv
   $ source venv/bin/activate

.. note::
   While hard to recommend, it is possible to install Python
   packages system-wide and use these instead of the virtual environment.

Pip-based Installation
----------------------

Use pip to install CloudnetPy into the virtual environment:

.. code-block:: console
		
   (venv)$ pip3 install cloudnetpy

CloudnetPy is now ready for use from that virtual environment.

.. note::

   CloudnetPy codebase is rapidly developing and the PyPI package does not
   necessarily contain all the latest features and modifications. To get an up-to-date
   version of CloudnetPy, download it directly from `GitHub
   <https://github.com/tukiains/cloudnetpy>`_.


