# CloudnetPy

[![Build Status](https://travis-ci.org/tukiains/cloudnetpy.svg?branch=master)](https://travis-ci.org/tukiains/cloudnetpy)
[![PyPI version](https://badge.fury.io/py/cloudnetpy.svg)](https://badge.fury.io/py/cloudnetpy)


CloudnetPy is a Python software for producing vertical profiles of cloud properties from ground-based remote sensing measurements. The Cloudnet processing combines cloud radar, optical lidar, microwave radiometer and model data. The measurements and model data are brought into common grid and classified as ice, liquid, aerosol, insects, and so on. Then, geophysical products such as ice water content can be retrieved in further processing steps.

CloudnetPy is a refactored fork of the currently operational (Matlab) processing code. The Python version will eventually feature several revised methods, extensive documentation, and more.

The documentation for CloudnetPy can be found at: https://cloudnetpy.readthedocs.io/en/latest/

<img src="docs/source/_static/20190423_mace-head_classification.png">

Installation
------------

CloudnetPy can be installed from the Python Package Index, PyPI, using pip:
```
pip3 install cloudnetpy
```

Links
-----

- Cloudnet website: http://devcloudnet.fmi.fi
