Developer's Guide
=================

CloudnetPy is hosted by Finnish Meteorological Institute (FMI) and
used to process cloud remote sensing data within ACTRIS research
infrastructure. We are happy to welcome the cloud remote sensing community
to provide improvements in the methods and their implementations, writing
tests and finding bugs.

How to contribute
-----------------

The right way to report a bug or obviously incorrect behaviour is
to open an issue on `CloudnetPy's Github page <https://github.com/tukiains/cloudnetpy/issues>`_.
Describe the problem and show the steps to reproduce it. Suggest a solution if you can but
this is not necessary. We will implement fixes as soon as possible into the
source code and releases.

However, any suggestion that affects the *methodology*, and thus the outcome of the
processing, needs to be carefully reviewed by experts of the ACTRIS cloud
remote sensing community. The bigger the modification, the bigger the validation
work needed before the actual implementation can happen. Nevertheless, don't be
afraid to present your development ideas! There is no such thing as perfect code,
and the same applies to scientific methods.


Testing
-------

To run Cloudnetpy tests, you first need to
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
    $ python3.7 -m venv venv
    $ source venv/bin/activate
    (venv) $ pip3 install numpy scipy netCDF4 matplotlib requests pytest

Running all tests
.................

Go to the tests folder and try to run all the tests:

.. code-block:: console

    (venv) $ cd tests
    (venv) $ python3.7 run_example_processing_and_tests.py

You should see the testing script loading some input files and starting
to run the processing and tests:

.. code-block:: console


    Loading input files...
      Done.
    ============================================================================================ test session starts =============================================================================================
    platform linux -- Python 3.6.8, pytest-3.3.2, py-1.5.2, pluggy-0.6.0
    rootdir: /home/tukiains/Documents/PYTHON/cloudnetpy, inifile:
    collected 96 items

    meta/test_raw.py ..                                                                                                                                                                                    [100%]

    ============================================================================================ 94 tests deselected =============================================================================================
    ================================================================================== 2 passed, 94 deselected in 0.27 seconds ===================================================================================
    ============================================================================================ test session starts =============================================================================================
    platform linux -- Python 3.6.8, pytest-3.3.2, py-1.5.2, pluggy-0.6.0
    rootdir: /home/tukiains/Documents/PYTHON/cloudnetpy, inifile:
    collected 96 items

    meta/test_raw.py ..                                                                                                                                                                                    [100%]

    ============================================================================================ 94 tests deselected =============================================================================================
    ================================================================================== 2 passed, 94 deselected in 0.26 seconds ===================================================================================
    /home/tukiains/Documents/PYTHON/cloudnetpy/venv/lib/python3.7/site-packages/numpy/core/fromnumeric.py:734: UserWarning: Warning: 'partition' will ignore the 'mask' of the MaskedArray.
    a.partition(kth, axis=axis, kind=kind, order=order)
    ============================================================================================ test session starts =============================================================================================
    platform linux -- Python 3.6.8, pytest-3.3.2, py-1.5.2, pluggy-0.6.0
    rootdir: /home/tukiains/Documents/PYTHON/cloudnetpy, inifile:
    collected 96 items

    meta/test_calibrated.py ..........                                                                                                                                                                     [ 22%]
    meta/test_products.py ..................................                                                                                                                                               [100%]


And so on.


.. note::

   Cloudnetpy performs high-level, sophisticated scientific processing. Most of the
   Cloudnetpy's low-level functions are unit tested, but it is notoriously
   difficult to write unambiguous tests for all high-level API calls (yet we
   *have* included at least rough tests for these). How well
   our classification scheme works with all
   possible instrument combinations? Or how accurate is our retrieved ice water
   content compared to the reality? These kind of questions can be finally
   answered only through rigorous scientific validation.


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





