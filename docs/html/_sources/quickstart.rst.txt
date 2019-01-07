==========
Quickstart
==========

CloudnetPy is available from `PyPI
<https://pypi.org/project/cloudnetpy/>`_, the Python package index. It allows a pip-based
installation.

First, install prerequisite software (if you already haven't):

.. code-block:: console
		
   $ sudo apt update && sudo apt upgrade
   $ sudo apt install python3 python3-pip libnetcdf-dev 

Then, create a new virtual enviroment and activate it:

.. code-block:: console
		
   $ python3 -m venv venv
   $ source venv/bin/activate

Install the required Python packages:

.. code-block:: console
		
   $ pip3 install scipy netcdf4 cloudnetpy

That's it! If you have cloud radar, ceilometer, microwave
radiometer and model data in NetCDF files, it's easy to
start processing. For example:

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

Note that the code base is rapidly developing and the PyPI package does not
contain all the latest features. To get an up-to-date version of
CloudnetPy, download it directly from `GitHub
<https://github.com/tukiains/cloudnetpy>`_:

.. code-block:: console

	$ git clone https://github.com/tukiains/cloudnetpy
