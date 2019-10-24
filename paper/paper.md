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
products such as ice water content, liquid water content and drizzle 
properties.

Methodology and prototype software to combine these different data sources, 
and to retrieve target classification and other products, were developed within 
the so-called Cloudnet project [@IllingworthEtAl07]. Since 2002, a network 
of around 15 stations around Europe has been established to collect and 
process Cloudnet data in a semi-operational manner. In the forthcoming years, 
Cloudnet will be an essential part of the research infrastructure ACTRIS 
(Aerosol, Clouds and Trace Gases Research Infrastructure), as the Cloudnet 
framework will be implemented in the cloud remote sensing component of ACTRIS.
ACTRIS is moving into implementation phase in 2020 and is supposed to be 
fully operational in 2025.

CloudnetPy is a Python implementation of the Cloudnet processing scheme. 
CloudnetPy covers full Cloudnet processing chain starting from the raw 
measurements and providing the same higher level products as the original 
prototype software. The outputs of these two software are similar but not 
identical, because many of the methods are revised and bugs fixed.
The API is designed to serve the operational processing, but also 
keeping in mind the scientific community which develops new methods 
and products.
 
# Acknowledgements

We thank..

# References