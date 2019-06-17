========
Overview
========

Cloudnet processing produces vertical profiles of cloud properties from ground-based remote sensing measurements.
Cloud radar, optical lidar, microwave radiometer and thermodynamical (model or radiosonde) data are combined to accurately
characterize clouds up to 15 km with high temporal and vertical resolution.

.. figure:: _static/example_data.png
	   :width: 500 px
	   :align: center

           Example measurements.
	   
The measurement and model data are brought into common grid and classified as ice, liquid, aerosol, insects, and so on.
Then, geophysical products such as ice water content can be retrieved in further processing steps.

CloudnetPy is a refactored fork of the currently operational (Matlab) processing code. It features
several revised methods, open source codebase, netCDF4 file format and extensive documentation.

Cloudnet processing scheme is eventually going to be part of the ACTRIS
research infrastructure which is moving into implementation phase in 2019. Operational
Cloudnet processing will have CloudnetPy in its core but additionally include a
calibration database and comprehensive quality control / assurance procedures:

.. figure:: _static/CLU_workflow.png
	   :width: 650 px
	   :align: center

           Workflow of operational Cloudnet processing in ACTRIS.


See also:

- Cloudnet home: http://devcloudnet.fmi.fi/
- CloudnetPy source: https://github.com/tukiains/cloudnetpy
- ACTRIS home: http://actris.eu/
- ACTRIS data portal: http://actris.nilu.no/
