from setuptools import find_packages, setup

version: dict = {}
with open("cloudnetpy/version.py", encoding="utf8") as f:
    exec(f.read(), version)  # pylint: disable=W0122

with open("README.md", encoding="utf8") as f:
    readme = f.read()

setup(
    name="cloudnetpy",
    version=version["__version__"],
    description="Python package for Cloudnet processing",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Finnish Meteorological Institute",
    author_email="actris-cloudnet@fmi.fi",
    url="https://github.com/actris-cloudnet/cloudnetpy",
    license="MIT License",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.10",
    setup_requires=["wheel"],
    install_requires=[
        "scipy",
        "netCDF4",
        "matplotlib",
        "requests",
        "cloudnetpy_qc>=1.4.1",
        "scikit-image",
        "rpgpy>=0.12.1",
        "toml",
    ],
    extras_require={
        "test": ["pytest", "pytest-flakefinder", "pylint", "mypy", "types-requests", "types-toml"],
        "dev": ["pre-commit"],
        "extras": ["voodoonet"],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
