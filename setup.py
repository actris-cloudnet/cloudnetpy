
from setuptools import setup, find_packages

version = {}
with open("cloudnetpy/version.py") as f:
    exec(f.read(), version)

with open('README.md') as f:
    readme = f.read()

setup(
    name='cloudnetpy',
    version=version['__version__'],
    description='Python package for Cloudnet processing',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Finnish Meteorological Institute',
    author_email='actris-cloudnet@fmi.fi',
    url='https://github.com/actris-cloudnet/cloudnetpy',
    license='MIT License',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    install_requires=['scipy', 'netCDF4', 'matplotlib', 'requests', 'pytz', 'pytest',
                      'cloudnetpy_qc>=0.0.4', 'scikit-image'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
