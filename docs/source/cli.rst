==================
Command-line usage
==================

CloudnetPy ships with a ``cloudnetpy`` console script that fetches
data from the `Cloudnet data portal <https://cloudnet.fmi.fi>`_, runs
the requested processing steps, and writes the resulting netCDF files
(and optionally plots) to disk. It is the fastest way to test
CloudnetPy or to reproduce a product for a given site and date without
writing any Python code.

For each requested product, the CLI processes only that step locally
and downloads the inputs (raw files or already-processed lower-level
products) directly from the data portal. Request a longer chain to
regenerate more of the pipeline yourself.

.. note::

    The CLI only works for sites and dates available in the Cloudnet
    data portal. For local files or for custom processing, use the
    :doc:`Python API <quickstart>` instead.

Basic usage
-----------

Specify a site, a date, and one or more products. CloudnetPy downloads
the required input files into ``./input/`` and writes the outputs into
``./output/``.

For instrument-level products, the CLI downloads the raw measurements
from the portal and runs the corresponding ``*2nc`` converter:

.. code-block:: console

    $ cloudnetpy --site munich --date 2023-07-29 --products radar

Multiple products can be processed in a single call:

.. code-block:: console

    $ cloudnetpy -s munich -d 2023-07-29 -p radar,lidar,mwr

If the date is omitted, today's date is used. The full list of valid
sites and products is fetched from the data portal at runtime; pass
``--help`` to see all options.

Selecting an instrument
-----------------------

When a site has more than one instrument that can produce a given
product, the CLI prompts you to choose. You can select up front using
either an instrument identifier:

.. code-block:: console

    $ cloudnetpy -s lindenberg -d 2024-06-01 -p radar -i radar:mira-35

or a persistent identifier (PID) for a specific physical device:

.. code-block:: console

    $ cloudnetpy -s lindenberg -d 2024-06-01 -p radar \
        -i radar:https://hdl.handle.net/21.12132/3.d6cc3d73f9dd4d4b

The same can be expressed compactly inside the product list:

.. code-block:: console

    $ cloudnetpy -s lindenberg -d 2024-06-01 -p 'radar[mira-35],lidar[cl61d]'

The ``-i`` flag can be given multiple times to set preferences for
several products at once.

Generating L2 products
----------------------

L2 products such as ``classification``, ``iwc``, ``lwc`` and
``drizzle`` are derived from the categorize file. Requesting only the
L2 product downloads the *existing* categorize file from the portal
and runs the L2 step locally:

.. code-block:: console

    $ cloudnetpy -s munich -d 2023-07-29 -p classification

Several L2 products can be derived from the same categorize file in
one command:

.. code-block:: console

    $ cloudnetpy -s munich -d 2023-07-29 -p classification,iwc,lwc,drizzle

Generating the categorize file locally
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To rebuild the categorize file yourself, add ``categorize`` to the
product list. The CLI downloads the L1b radar, lidar, MWR and model
files from the portal and runs ``generate_categorize`` locally. The
L2 step then uses the freshly-generated categorize file:

.. code-block:: console

    $ cloudnetpy -s munich -d 2023-07-29 -p categorize,classification

To regenerate the L1b inputs from raw data as well, add the
instrument-level products to the chain:

.. code-block:: console

    $ cloudnetpy -s munich -d 2023-07-29 \
        -p radar,lidar,mwr,categorize,classification

Options for the categorize step are passed as JSON via ``--options``:

.. code-block:: console

    $ cloudnetpy -s munich -d 2023-07-29 -p categorize \
        --options '{"temperature_offset": 3}'

A specific weather model can be selected with ``--model``:

.. code-block:: console

    $ cloudnetpy -s munich -d 2023-07-29 -p categorize --model ecmwf

Plotting
--------

Generate PNG plots of the processed files alongside the netCDF
outputs:

.. code-block:: console

    $ cloudnetpy -s munich -d 2023-07-29 -p classification --plot

Use ``--show`` to open the plots in a window instead of (or in
addition to) saving them. Restrict which variables to plot with
``--variables``:

.. code-block:: console

    $ cloudnetpy -s munich -d 2023-07-29 -p radar --plot \
        --variables Zh,v

Other useful flags
------------------

================================  ==========================================
``--input PATH``                  Directory for downloaded raw files
                                  (default ``./input/``).
``--output PATH``                 Directory for processed files
                                  (default ``./output/``).
``--dl``                          Download raw data only; do not process.
``--force-download``              Re-download raw files even if they exist
                                  locally.
``-h``, ``--help``                Show all options.
================================  ==========================================

Output layout
-------------

Files are organized by site, date, and processing level:

.. code-block:: text

    output/
    └── munich/
        └── 2023-07-29/
            ├── instrument/
            │   ├── 20230729_munich_chm15k_<pid>.nc
            │   ├── 20230729_munich_hatpro_<pid>.nc
            │   ├── 20230729_munich_mira-35_<pid>.nc
            │   └── 20230729_munich_ecmwf.nc
            └── geophysical/
                ├── 20230729_munich_categorize.nc
                └── 20230729_munich_classification.nc

The same layout is used for raw input files under ``input/``.
