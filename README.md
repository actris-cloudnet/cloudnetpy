# CloudnetPy

[![CloudnetPy CI](https://github.com/actris-cloudnet/cloudnetpy/actions/workflows/test.yml/badge.svg)](https://github.com/actris-cloudnet/cloudnetpy/actions/workflows/test.yml)
[![PyPI version](https://badge.fury.io/py/cloudnetpy.svg)](https://badge.fury.io/py/cloudnetpy)
[![DOI](https://zenodo.org/badge/233602651.svg)](https://zenodo.org/badge/latestdoi/233602651)
[![status](https://joss.theoj.org/papers/959971f196f617dddc0e7d8333ff22b7/status.svg)](https://joss.theoj.org/papers/959971f196f617dddc0e7d8333ff22b7)

CloudnetPy is Python software designed for producing vertical profiles of cloud properties from ground-based
remote sensing measurements. The Cloudnet processing combines data from cloud radar, optical lidar,
microwave radiometer, and numerical weather prediction models. Measurements and model data are brought
into a common grid and classified as ice, liquid, aerosol, insects, and so on. Subsequently, geophysical
products such as ice water content can be retrieved.

![CloudnetPy example output](https://raw.githubusercontent.com/actris-cloudnet/cloudnetpy/main/docs/source/_static/20230831_lindenberg_classification-9b74f4ac-target_classification.png)

## Installation Steps
### Option 1: Install via pip
To install CloudnetPy, use pip:
```bash
pip install cloudnetpy
```

### Option 2: Manual Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/actris-cloudnet/cloudnetpy.git
cd cloudnetpy
pip install -r requirements.txt
```

### Verification
To verify the installation:
```bash
python -m cloudnetpy --help
```

## Documentation
Comprehensive documentation is available at:
- [CloudnetPy Documentation](https://cloudnetpy.readthedocs.io/)

## Contributing
Contributions are welcome! Please check the [CONTRIBUTING.md](https://github.com/actris-cloudnet/cloudnetpy/blob/main/CONTRIBUTING.md) for guidelines.

## Version History
Details on release versions and changes are available in the [CHANGELOG](https://github.com/actris-cloudnet/cloudnetpy/blob/main/CHANGELOG.md).

## Help and Support
For support and common issues, refer to:
- [GitHub Issues](https://github.com/actris-cloudnet/cloudnetpy/issues)

## Citation
If you use CloudnetPy in your work, please use the following citation:
[![DOI](https://zenodo.org/badge/233602651.svg)](https://zenodo.org/badge/latestdoi/233602651)

## License

MIT
