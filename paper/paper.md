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
possible to classify atmospheric targets and retrieve geophysical 
products such as ice water content, liquid water content and drizzle 
properties.

Methodology and software to combine these different data sources, and to 
retrieve target classification and other products, were developed within 
the so-called Cloudnet project. A network of around 15 stations around 
Europe has been established since 2002 to collect data and test methods 
in operative manner.

CloudnetPy is a Python implementation of the Cloudnet processing scheme.


# Acknowledgements

We thank..

# References