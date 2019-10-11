========
Overview
========

Cloudnet processing produces vertical profiles of cloud properties from ground-based remote sensing measurements.
Cloud radar, optical lidar, microwave radiometer and thermodynamical (model or radiosonde) data are combined to accurately
characterize clouds up to 15 km with high temporal and vertical resolution.

.. figure:: _static/example_data.png
	   :width: 500 px
	   :align: center

           Example input data, part of it, used in Cloudnet processing: Radar reflectivity factor (top), mean
           doppler velocity (2nd), lidar backscatter coefficient (3rd),
           and liquid water path from microwave radiometer (bottom).
	   
The measurement and model data are brought into common grid and classified as ice, liquid, aerosol, insects, and so on.
Then, geophysical products such as ice water content can be retrieved in further processing steps.
A more detailed description can be found in `Illingworth 2007`_ and references in it.

.. _Illingworth 2007: https://journals.ametsoc.org/doi/abs/10.1175/BAMS-88-6-883

.. important::

   CloudnetPy is a refactored fork of the currently operational (Matlab) processing code. CloudnetPy features
   several revised methods and bug fixes, open source codebase, netCDF4 file format and extensive documentation.

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
