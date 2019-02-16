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

Install cloudnetpy into the virtual environment:

.. code-block:: console
		
   (venv)$ pip3 install cloudnetpy

That's it! Processing is easy using CloudnetPy's high level APIs.

For example, let's convert a raw METEK MIRA-36 netCDF file into
Cloudnet netCDF file that can be used in further processing steps.

.. code-block:: python

    from cloudnetpy.mira import mira2nc
    mira2nc('raw_radar.mmclx', 'radar.nc', {'name': 'Mace-Head'})

In the next example we create a categorize file from already
calibrated measurement files.

.. code-block:: python

   from cloudnetpy.categorize import generate_categorize
   input_files = {
       'radar': 'radar.nc',
       'lidar': 'lidar.nc',
       'model': 'model.nc',
       'mwr': 'mwr.nc'
       }
   output_file = 'test.nc'
   generate_categorize(input_files, output_file)

Note that the CloudnetPy codebase is rapidly developing and the PyPI package does not
contain all the latest features and modifications. To get an up-to-date
version of CloudnetPy, download it directly from `GitHub
<https://github.com/tukiains/cloudnetpy>`_:

.. code-block:: console

	$ git clone https://github.com/tukiains/cloudnetpy

