# CloudnetPy

[![CloudnetPy CI](https://github.com/actris-cloudnet/cloudnetpy/actions/workflows/test.yml/badge.svg)](https://github.com/actris-cloudnet/cloudnetpy/actions/workflows/test.yml)
[![Documentation Status](https://readthedocs.org/projects/cloudnetpy/badge/?version=latest)](https://cloudnetpy.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/cloudnetpy.svg)](https://badge.fury.io/py/cloudnetpy)
[![DOI](https://zenodo.org/badge/233602651.svg)](https://zenodo.org/badge/latestdoi/233602651)
[![status](https://joss.theoj.org/papers/959971f196f617dddc0e7d8333ff22b7/status.svg)](https://joss.theoj.org/papers/959971f196f617dddc0e7d8333ff22b7)

CloudnetPy is Python software designed for producing vertical profiles of cloud properties from ground-based
remote sensing measurements. The Cloudnet processing combines data from cloud radar, optical lidar,
microwave radiometer, and numerical weather prediction models.
Measurements and model data are brought into a common grid and
classified as ice, liquid, aerosol, insects, and so on.
Subsequently, geophysical products such as ice water content can be
retrieved in further processing steps. See [Illingworth et al. (2007)](https://doi.org/10.1175/BAMS-88-6-883) for more details about the concept.

CloudnetPy is a rewritten version of the original Cloudnet Matlab code. It features several revised methods, extensive documentation, and more.

- CloudnetPy documentation: https://cloudnetpy.readthedocs.io/en/latest/
- Cloudnet data portal: https://cloudnet.fmi.fi

![CloudnetPy example output](https://raw.githubusercontent.com/actris-cloudnet/cloudnetpy/main/docs/source/_static/20230831_lindenberg_classification-9b74f4ac-target_classification.png)

## Installation

### From PyPI

```
python3 -m pip install cloudnetpy
```

### From the source

```sh
git clone https://github.com/actris-cloudnet/cloudnetpy
cd cloudnetpy/
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install .
```

## Citing

If you wish to acknowledge CloudnetPy in your publication, please cite:

> Tukiainen et al., (2020). CloudnetPy: A Python package for processing cloud remote sensing data. Journal of Open Source Software, 5(53), 2123, https://doi.org/10.21105/joss.02123

## Contributing

We encourage you to contribute to CloudnetPy! Please check out the [contribution guidelines](CONTRIBUTING.md) about how to proceed.

## Development

Follow the installation instructions from the source above but install with the development dependencies and [pre-commit](https://pre-commit.com/) hooks:

```sh
python3 -m pip install -e .[dev,test]
pre-commit install
```

Run unit tests:

```sh
python3 -m pytest --flake-finder --flake-runs=2
```

Run single unit test:

```sh
python3 -m pytest tests/unit/test_hatpro.py
```

Run end-to-end tests:

```sh
python3 tests/e2e_test.py
```

```sh
for f in cloudnetpy/model_evaluation/tests/e2e/*/main.py; do $f; done
```

Force `pre-commit` checks (`ruff`, `mypy`, etc.) for all files:

```sh
pre-commit run --all
```

## License

MIT
