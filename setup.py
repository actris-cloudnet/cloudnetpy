
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='cloudnetpy',
    version='0.1.0',
    description='Python package for Cloudnet processing',
    long_description=readme,
    author='Simo Tukiainen',
    author_email='simo.tukiainen@fmi.fi',
    url='https://github.com/tukiains/cloudnetpy',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    python_requires='>=3.6',
    install_requires=['scipy>=1.1', 'netCDF4>=1.4.1'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
