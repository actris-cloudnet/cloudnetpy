
==========
Input data
==========

A Cloudnet site contains three main instruments:

- Cloud radar
- Optical lidar, typically ceilometer
- Microwave radiometer

In addition, height-resolved model data (temperature, pressure, etc.) are
needed for the processing.

Raw measurements need to be first processed (SNR screened, etc.) with some
separate software and stored as netCDF files. Specifications for the
input files are in this document (?)
 

Cloud radar
===========

Sub-millimeter cloud radar is the key instrument in Cloudnet data processing.
Cloud radars emit electromagnetic waves and detect radiation, backscattered from
atmospheric particles such as ice crystals, insects and birds. From the scattered
signal, the so-called *Doppler spectra* for different time delays, or range gates,
corresponding to different altitudes, can be calculated.
Other variables such as *radar reflectivity*, *Doppler velocity* and *width* can be
furthermore derived from the radar spectra.

Most modern cloud radars are also polarimetric. Polarization measurements highly
improve chances to distinguish different type of scatterers from the data.

In the standard Cloudnet operation mode, cloud radar is pointing vertically and
measures continuously with high temporal resolution.

.. figure:: _static/radar.jpg
	   :width: 300 px
	   :align: center

           MIRA-36 Cloud Radar
		   
Frequency
---------

Cloud radars typically operate on around 35 GHz or 94 GHz. Cloud radars at 35 GHz have
a larger antenna and are typically bigger, more powerful (and more expensive)
than 94 GHz radars.

Temporal resolution
-------------------

Temporal sampling resolution of a cloud radar can be anything between 1 and 30 s.
Typical values are 1, 10 and 30 s. However, arbitrary or regular gaps in the data
are possible due to e.g. horizontal scanning or other off-zenith measurements.


Optical lidar
=============

Cloudnet processing uses low-powered optical lidar to find liquid
cloud bases, precipitation supercooled liquids, and aerosols,
which are not sensitive to the radar. This is typically a ceilometer
but it can also be a Doppler lidar.

.. figure:: _static/dlidar.jpg
	   :width: 300 px
	   :align: center

           Doppler lidar, from HALO-photonics.



Microwave radiometer
====================











