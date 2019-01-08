========
Overview
========

Cloudnet processing produces vertical profiles of cloud properties from ground-based remote sensing measurements.
Cloud radar, optical lidar, microwave radiometer and model data are combined in order to characterize
clouds up to 15 km with high temporal and vertical resolution.

.. figure:: _static/example_data.jpg
	   :width: 400 px
	   :align: center

           Example measurements.
	   
The measurements and model data are brought into common grid and classified as ice, liquid, aerosol, insects, and so on.
Then, geophysical products such as ice water content can be retrieved in further processing steps.

CloudnetPy is a refactored fork of the currently operational (Matlab) processing code. It features, e.g.,
several revised methods, open source code base, netCDF4 file format, and extensive documentation.

The Cloudnet processing scheme is eventually going to be part of the ACTRIS
reseach infrastructure. ACTRIS is moving into implementation phase in 2019.

See also:

- Cloudnet home: http://devcloudnet.fmi.fi/
- CloudnetPy source: https://github.com/tukiains/cloudnetpy
- ACTRIS home: http://actris.eu/
- ACTRIS data portal: http://actris.nilu.no/
