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
date: 24 October 2019
bibliography: paper.bib
---

# Summary

Active ground-based remote sensing instruments provide vertical profiles of 
clouds and aerosols with high vertical and temporal resolution. Cloud radars 
typically operate in the sub-millimeter wavelength region, around 35 or 94 GHz, 
mainly providing information on ice clouds, rain and insects. Optical lidars,
on the other hand, are more sensitive to liquid clouds and aerosols. 
Combining the two supplementary instruments with model temperature makes 
it possible to classify the various scattering objects in the atmosphere.
Furthermore, adding a passive microwave radiometer, an instrument measuring 
liquid water path, allows quantitative retrievals of geophysical 
products such as ice water content, liquid water content 
and drizzle properties [@OConnorEtAl05, @HoganEtAl06].

Methodology and prototype software to combine these different data sources, 
and to retrieve target classification and other products, were developed within 
the so-called Cloudnet project [@IllingworthEtAl07]. Since the project started 
in 2002, a network of around 15 stations around Europe has been established 
to regularly collect, process and distribute Cloudnet data (http://cloudnet.fmi.fi). 
While the methodology is validated, more robust and operational processing 
software is called for. In the forthcoming years, Cloudnet will be one of 
the key components in ACTRIS (Aerosol, Clouds and Trace Gases Research 
Infrastructure) [@ACTRIS_handbook], where Cloudnet framework 
is used to process gigabytes of data per day in near real time. 
ACTRIS is moving into implementation phase in 2020 and is supposed 
to be fully operational in 2025.

CloudnetPy is a Python implementation of the Cloudnet processing scheme. 
CloudnetPy covers the full Cloudnet processing chain starting from the raw 
measurements and providing roughly the same functionality as the original 
Cloudnet software written in Matlab and C. The outputs of the two programs 
are similar but not identical because several methods were revised 
during the refactoring process. For example, because most modern cloud 
radars are polarimetric, CloudnetPy uses linear depolarization ratio 
to improve the detection of melting layer and insects. Also the 
liquid layer detection is now based on the shape of lidar backscatter 
profile instead of threshold values [@TuononenEtAl19]. Detailed
verification of the updated methods is a subject of future studies.
The CloudnetPy API is designed to serve the operational processing, but 
it should be straightforward for site operators and scientific community
with access to raw data to run the software, improve existing 
methods and develop new products.


# Acknowledgements

The authors would like to thank the Academy of Finland for supporting
the development of ACTRIS activities in Finland.

# References