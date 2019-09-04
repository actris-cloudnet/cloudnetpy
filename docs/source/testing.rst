Testing
=======

To run Cloudnetpy tests, you first need to
clone the whole repository from `GitHub
<https://github.com/tukiains/cloudnetpy>`_:

.. code-block:: console

	$ git clone https://github.com/tukiains/cloudnetpy

Testing environment
-------------------

Now, create a virtual environment and install external packages
needed by Cloudnetpy:

.. code-block:: console

    $ cd cloudnetpy
    $ python3.7 -m venv venv
    $ source venv/bin/activate
    (venv) $ pip3 install numpy scipy netCDF4 matplotlib requests pytest

Running all tests
-----------------

Go to the tests folder and try to run all the tests:

.. code-block:: console

    (venv) $ cd tests
    (venv) $ python3.7 test.py

You should see the testing script loading some input files and starting
to run the tests:

.. code-block:: console


    ###################### Running all CloudnetPy tests ######################

    Loading input files...    Done.

    Testing misc CloudnetPy routines:

    ========================== test session starts ========================
    platform linux -- Python 3.7.3, pytest-5.1.2, py-1.8.0, pluggy-0.12.0
    rootdir: /home/tukiains/Temp/cloudnetpy
    collected 29 items

    ../cloudnetpy/tests/test_utils.py .............................                 [100%]

    ========================== 29 passed in 0.06s ==========================
    ========================== test session starts =========================
    platform linux -- Python 3.7.3, pytest-5.1.2, py-1.8.0, pluggy-0.12.0
    rootdir: /home/tukiains/Temp/cloudnetpy
    collected 9 items

    ../cloudnetpy/categorize/tests/test_atmos.py .........                          [100%]


And so on. If the tests complete succesfully, you should see in the end:

.. code-block:: console

    ############ All tests passed and processing works correctly! ############

Notes
-----

Cloudnetpy performs high-level, complicated scientific processing. Most of the
Cloudnetpy's low-level functions are unit tested, but it is notoriously
difficult to write unambiguous tests for the high-level API calls.
How well our classification scheme works with different kind of instrumentation
and site characteristics? Or how accurate is our ice water content product
compared to the reality? These kind of questions can be answered only through
rigorous scientific validation. Nevertheless, our goal is at least roughly
check the correctness of the retrieved quantities.





