==========
Quickstart
==========

CloudnetPy is available from `PyPI
<https://pypi.org/project/cloudnetpy/>`_, the Python package index. It allows a pip-based
installation. CloudnetPy requires Python 3.7 or newer.

First, install prerequisite software (if you already haven't):

.. code-block:: console
		
   $ sudo apt update && sudo apt upgrade
   $ sudo apt install python3.7 python3.7-venv python3-pip

Then, create a new virtual environment and activate it:

.. code-block:: console
		
   $ python3.7 -m venv venv
   $ source venv/bin/activate

Install cloudnetpy:

.. code-block:: console
		
   (venv)$ pip3 install cloudnetpy

That's it! If you have cloud radar, ceilometer, microwave
radiometer and model data in NetCDF files, it's easy to
start processing using CloudnetPy's high level APIs.
For example:

.. code-block:: python

   from cloudnetpy import categorize as cat
   
   input_files = (
		 'radar_file.nc',
		 'lidar_file.nc',
		 'mwr_file.nc',
		 'model_file.nc'
		 )
   output_file = 'test.nc'

   cat.generate_categorize(input_files, output_file)

Note that the CloudnetPy codebase is rapidly developing and the PyPI package does not
contain all the latest features and modifications. To get an up-to-date
version of CloudnetPy, download it directly from `GitHub
<https://github.com/tukiains/cloudnetpy>`_:

.. code-block:: console

	$ git clone https://github.com/tukiains/cloudnetpy

