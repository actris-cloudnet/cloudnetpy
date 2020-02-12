
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
    author_email='simo.tukiainen@fmi.fi',
    url='https://github.com/tukiains/cloudnetpy',
    license='MIT License',
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=['numpy>=1.16', 'scipy>=1.2', 'netCDF4>=1.4.2',
                      'matplotlib>=3.0.2', 'requests>=2.21'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
)
