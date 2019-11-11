Developer's Guide
=================

CloudnetPy is hosted by Finnish Meteorological Institute (FMI) and
will be eventually used to process cloud remote sensing data in the
ACTRIS research infrastructure. We are happy to welcome the cloud remote sensing community
to provide improvements in the methods and their implementations, writing
tests and fixing bugs.

How to contribute
-----------------

Instructions can be found from `CloudnetPy's Github page <https://github.com/tukiains/cloudnetpy/blob/master/CONTRIBUTING.md>`_.


Testing
-------

To run CloudnetPy tests, you first need to
clone the whole repository from `GitHub
<https://github.com/tukiains/cloudnetpy>`_:

.. code-block:: console

	$ git clone https://github.com/tukiains/cloudnetpy

Testing environment
...................

Now, create a virtual environment and install external packages
needed by Cloudnetpy:

.. code-block:: console

    $ cd cloudnetpy
    $ python3 -m venv venv
    $ source venv/bin/activate
    (venv) $ pip3 install numpy scipy netCDF4 matplotlib requests pytest

Running all tests
.................

Go to the tests folder and execute the script that runs a full Cloudnet
processing for example files and all tests:

.. code-block:: console

    (venv) $ cd tests
    (venv) $ python3 run_example_processing_and_tests.py

You should see the testing script loading some input files and starting
to run the processing and tests:

.. code-block:: console

 ============================= test session starts ==============================
 platform linux -- Python 3.7.1, pytest-4.1.1, py-1.7.0, pluggy-0.8.1
 rootdir: /home/tukiains/Documents/PYTHON/cloudnetpy, inifile:
 plugins: cov-2.8.1
 collected 265 items

 unit/test_atmos.py ............
 unit/test_ceilo.py ....
 unit/test_ceilometer.py ........
 unit/test_classify.py .........
 unit/test_cloudnetarray.py ........
 unit/test_datasource.py ........
 unit/test_drizzle.py ..........
 unit/test_droplet.py .................
 unit/test_falling.py ..............
 unit/test_freezing.py ...
 unit/test_insects.py ......
 unit/test_lidar.py ..
 unit/test_melting.py ............
 unit/test_meta_for_old_files.py ..
 unit/test_mira.py .
 unit/test_model.py ........
 unit/test_mwr.py ...
 unit/test_output.py .........
 unit/test_plotting.py .............
 unit/test_product_tools.py ...
 unit/test_radar.py .........
 unit/test_rpg.py ...
 unit/test_utils.py ..............................................................................................
 unit/test_vaisala.py .......

 ========================== 265 passed in 0.87 seconds ==========================
 ============================= test session starts ==============================
 platform linux -- Python 3.7.1, pytest-4.1.1, py-1.7.0, pluggy-0.8.1 -- /home/tukiains/Documents/PYTHON/cloudnetpy/venv/bin/python3
 cachedir: .pytest_cache
 rootdir: /home/tukiains/Documents/PYTHON/cloudnetpy, inifile:
 plugins: cov-2.8.1
 collecting ... collected 4 items

 meta/test_metadata.py::test_variables PASSED
 meta/test_metadata.py::test_global_attributes PASSED
 meta/test_metadata.py::test_variable_units FAILED
 meta/test_metadata.py::test_attribute_values PASSED


And so on.


.. note::

   Cloudnetpy performs relatively complicated scientific processing, converting
   noisy measurement data into higher level products. Most of the
   Cloudnetpy's low-level functions are unit tested, but it is
   difficult to write unambiguous tests for the high-level API calls.
   However, the quality of the processed files can be at least roughly
   checked using CloudnetPy's quality control functions.


Coding guidelines
-----------------

- Use `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ standard.

- Check your code using, e.g., `Pylint <https://www.pylint.org/>`_.

- Write `Google-style docstrings <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.

- Follow `Google Python Style Guide <https://github.com/google/styleguide/blob/gh-pages/pyguide.md>`_.

- Write *short* functions and classes.

- Use *meaningful* names for variables, functions, etc.

- Write *minimal* amount of comments. Your code should be self-explaining.

- Always unit-test your code!

Further reading:

- `Clean Code <https://www.oreilly.com/library/view/clean-code/9780136083238/>`_
- `Clean Code in Python <https://www.packtpub.com/eu/application-development/clean-code-python>`_
- `The Pragmatic Programmer <https://pragprog.com/book/tpp20/the-pragmatic-programmer-20th-anniversary-edition>`_
