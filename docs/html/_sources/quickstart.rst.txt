==========
Quickstart
==========

CloudnetPy is available from PyPI, the Python package index. It allows a pip-based
installation.

Create new virtual enviroment and activate it:

.. code-block:: console
		
   $ python3 -m venv venv
   $ source venv/bin/activate

Install required packages:

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

However, note that the code base is rapidly developing and the PyPI package does not
contain all the latest features. To get up-to-date version of
CloudnetPy, download it directly from github:

.. code-block:: console

	$ git clone https://github.com/tukiains/cloudnetpy
