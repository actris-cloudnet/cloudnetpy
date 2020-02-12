---
title: 'CloudnetPy: A Python package for processing cloud remote sensing data'
tags:
  - Python
  - cloud radar
  - lidar
  - microwave radiometer
  - remote sensing
authors:
  - name: Simo Tukiainen
    orcid: 0000-0002-0651-4622
    affiliation: 1
  - name: Ewan O'Connor
    affiliation: 1
  - name: Anniina Korpinen
    affiliation: 1
affiliations:
 - name: Finnish Meteorological Institute, Helsinki, Finland
   index: 1
date: 12 February 2020
bibliography: paper.bib
---

# Summary

Active ground-based remote sensing instruments such as cloud radars and lidars 
provide vertical profiles of clouds and aerosols with high vertical and 
temporal resolution. Cloud radars typically operate in the sub-millimeter 
wavelength region, around 35 or 94 GHz, 
and are sensitive to clouds, particularly ice clouds, rain and insects. Lidars operating 
at visible and near-infrared wavelengths 
on the other hand, are more sensitive to liquid clouds and aerosols. 
Combining these two complementary data sources with temperature and humidity profiles 
from a numerical weather prediction model or radiosonde makes it possible to accurately classify 
the various scattering hydrometeors in the atmosphere, diagnosing them as: rain drops, 
ice particles, melting ice particles, liquid droplets, supercooled liquid droplets, 
drizzle drops, insects and aerosol particles. 
Furthermore, adding a passive microwave radiometer, an instrument measuring 
liquid water path, attenuation corrections and quantitative retrievals of geophysical 
products such as ice water content, liquid water content 
and drizzle properties become feasible [@OConnorEtAl05, @HoganEtAl06].

Methodology and prototype software to combine these different data sources, 
and to retrieve target classification and other products, were developed within 
the EU-funded Cloudnet project [@IllingworthEtAl07]. Since Cloudnet started in 
2002, the network has expanded from 3 stations to a coordinated
and continuously operated network of around 15 stations across Europe. 
The network routinely collects, processes and distributes Cloudnet data (http://cloudnet.fmi.fi). 
While the current methodology has been validated, it is important to develop the Cloudnet software 
so that it can efficiently handle large amounts of data and reliably perform 
continuous data processing. In the forthcoming years, Cloudnet will be one of 
the key components in ACTRIS (Aerosol, Clouds and Trace Gases Research 
Infrastructure) [@ACTRIS_handbook], where the Cloudnet framework 
will process gigabytes of cloud remote sensing data per day 
in near real time. The ACTRIS RI is now in its implementation phase and 
aims to be fully operational in 2025. 

CloudnetPy is a Python implementation of the Cloudnet processing scheme. 
CloudnetPy covers the full Cloudnet processing chain starting from the raw 
measurements and providing similar functionality to the original, 
proprietary Cloudnet software written in Matlab and C. The output from CloudnetPy
is no longer identical to the original scheme because several methods have been 
revised and improved during the refactoring process. For example, as most modern cloud 
radars are polarimetric, CloudnetPy uses the linear depolarization ratio 
to improve the detection of the melting layer and insects. Liquid layer detection is 
now based on the lidar attenuated backscatter profile shape instead of relying only on
threshold values [@TuononenEtAl19]. Detailed 
verification of the updated methods is a subject of future studies. 
The CloudnetPy API is designed to serve the operational cloud remote sensing data 
processing in ACTRIS, but it will be straightforward for site operators and the 
scientific community with access to the raw data to run the software, improve 
existing methods and develop new products.


# Acknowledgements

The research was supported by the European Union Framework Programme for Research and 
Innovation, Horizon 2020 (ACTRIS-2,  grant  no.  654109). The authors would like to 
thank the Academy of Finland for supporting ACTRIS activities in Finland
and Lauri Kangassalo for providing comments on the manuscript.

# References
