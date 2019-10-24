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

Ground-based remote sensing measurements can provide vertically resolved 
cloud properties with high vertical and temporal resolution. Cloud radars 
typically operate in the sub-millimeter wavelength region around 35 or 94 GHz, 
mainly providing information on ice clouds, rain and insects. Optical lidars,
on the other hand, are more sensitive to liquid clouds, rain and drizzle. 
Combining these two instruments with a passive microwave radiometer, which 
gives an estimate of the integrated liquid water, and model data, makes it 
possible to classify atmospheric targets and to retrieve geophysical 
products such as ice water content [@HoganEtAl06], liquid water content 
and drizzle properties [@OConnorEtAl05].

Methodology and prototype software to combine these different data sources, 
and to retrieve target classification and other products, were developed within 
the so-called Cloudnet project [@IllingworthEtAl07]. Since 2002, a network 
of around 15 stations around Europe has been established to collect and 
process Cloudnet data in a semi-operational manner. In the forthcoming years, 
Cloudnet will be an essential part of the research infrastructure ACTRIS 
(Aerosol, Clouds and Trace Gases Research Infrastructure), as the Cloudnet 
framework is implemented in the cloud remote sensing component of ACTRIS.
ACTRIS is moving into implementation phase in 2020 and is supposed to be 
fully operational in 2025 [@ACTRIS_handbook].

CloudnetPy is a Python implementation of the Cloudnet processing scheme. 
CloudnetPy covers the full Cloudnet processing chain starting from the raw 
measurements and providing roughly the same functionality as the original 
software written in Matlab and C. The outputs of the two programs 
are similar but not identical because several methods were revised. 
For instance, because most modern cloud 
radars are polarimetric, CloudnetPy uses linear depolarization ratio 
to improve the detection of melting layer and insects. Also the 
liquid layer detection is now based on the shape of lidar backscatter 
profile instead of a simple threshold value [@TuononenEtAl19].

The CloudnetPy API is designed to serve the operational processing, but 
it should be straightforward for site operators and scientific community 
to run the software, improve existing methods and develop new products.
 
# Acknowledgements

The authors would like to thank the Academy of Finland for supporting
the development of ACTRIS activities in Finland.

# References