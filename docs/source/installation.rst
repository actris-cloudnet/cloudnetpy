=========================
Installation instructions
=========================

CloudnetPy can be installed on any computer having Python 3.10 or higher.
The actual installation procedure depends on the operating system. The
instructions below are for Ubuntu 22.04.

Python installation
-------------------

.. code-block:: console

   $ sudo apt update && sudo apt upgrade
   $ sudo apt install python3-venv python3-pip python3-tk

Virtual environment
-------------------

Create a new virtual environment and activate it:

.. code-block:: console

   $ python3 -m venv venv
   $ source venv/bin/activate


Pip-based installation
----------------------

CloudnetPy is available from Python Package Index, `PyPI
<https://pypi.org/project/cloudnetpy/>`_.
Use Python's package manager, `pip <https://pypi.org/project/pip/>`_,
to install CloudnetPy package into the virtual environment:

.. code-block:: console

   (venv)$ pip3 install cloudnetpy

CloudnetPy is now ready for use from that virtual environment.
