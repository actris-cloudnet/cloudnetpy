# Changelog

## 1.32.0 – 2022-05-13

- Skip corrupted profiles in CL51 files
- Add missing units
- Test Windows and macOS
- Test Python 3.10 support
- Fix code formatting

## 1.31.2 - 2022-03-23

- Cast `parsivel` metadata to floats

## 1.31.1 - 2022-03-22

- Allow `rpg-fmcw-94` latitude and longitude values to vary
- Raise error if only one valid `mira` timestamp

## 1.31.0 - 2022-03-14

- Drizzle without spectral width
- Minor fixes

## 1.30.2 - 2022-03-07

- Bug fix

## 1.30.1 - 2022-03-04

- Bug fixes

## 1.30.0 - 2022-03-04

- Adds support for `chm15k` ceilometers that contain `beta_att` instead of `beta_raw`
- Separates test-dependencies in `setup.py`
- Fixes myriad type hints

## 1.29.4 - 2022-02-15

`rpg-fmcw-94` fixes:

- Does not crash if elevation angle missing
- Sorts timestamps and removes duplicates

## 1.29.3 - 2022-02-09

- Adds screening of small `rpg-fmcw-94` width values.

## 1.29.2 - 2022-02-07

- Fix chm15k(x) source attribute

## 1.29.1 - 2022-02-02

- Add HALO Doppler lidar to `instruments.py`

## 1.29.0 - 2022-01-31

- Add Radiometrics support

## 1.28.1 - 2022-01-24

- Use the same plotting routines for current and legacy files

## 1.28.0 - 2022-01-17

Return dimensions of generated images

## 1.27.7 - 2021-12-21

- Adds timestamp sorting and duplicate removal for `mira`

## 1.27.6 - 2021-12-21

- Raises custom exception from bad model file

## 1.27.5 - 2021-12-20

- Removes duplicate `hatpro` timestamps

## 1.27.4 - 2021-12-19

- Sort and remove duplicates from Vaisala cl31/51 data

## 1.27.3 - 2021-12-16

- Correctly classify first layer as ice

## 1.27.2 - 2021-12-16

- Fixes bug in freezing region determination

## 1.27.1 - 2021-12-15

- Fix bug in model plotting

## 1.27.0 - 2021-12-15

- Lidar / radar data gaps removed from the time array
- Data gaps shown as vertical grey bars in classification / status plots
- Level 2 products harmonized against legacy files and netCDF validator

## 1.26.0 - 2021-12-09

- Improved melting layer detection
- Improved drizzle / insects classification
- Detection status variable to match legacy files
- Minor fixes to plotting and classification file metadata

## 1.25.1 - 2021-11-29

- Removes quality control from CloudnetPy package
- Adds speckle filter to BASTA data
- Removes classification results from profiles without any lidar data

## 1.25.0 - 2021-11-29

- Uses Python 3.8 and newer. Older Python versions are not supported from now on.
- Updates categorize file:
    - Fixes netCDF metadata
    - Interpolates lidar data using nearest neighbor
    - Adds 1px melting layer when not detected from data

## 1.24.0 - 2021-11-22

- Harmonized `hatpro` processing

## 1.23.2 - 2021-11-18

- Screening of `mira` files with deviating height vector
- Screening of `mira` profiles with deviating zenith angle
- snr limit option to `pollyxt2nc`

## 1.23.0 - 2021-11-16

- Refactored radar processing
- `instruments.py` module
- Minor fixes to global attributes
- Improved tests
- Deprecated `keep_uuid`  option

## 1.22.4 - 2021-11-09

- Fix processing of old `chm15k` ceilometers

## 1.22.3 - 2021-11-08

- Minor tuning of the ceilometer screening method

## 1.22.1 - 2021-11-05

- Improved ceilometer data screening

## 1.22.0 - 2021-10-30

- PollyXT support
- Harmonized lidar files

## 1.21.2 - 2021-10-09

- Avoid crash in `update_nc` if invalid nc file
- QC adjustment

## 1.21.1 - 2021-09-29

- Custom exception for disdrometer files that can not be read

## 1.21.0 - 2021-09-28

- Initial support for `Parsivel2` and `Thies-LNM` disdrometers
- Quality control adjustments

## 1.20.4 - 2021-09-21

- Adds more contrast to ice clouds

## 1.20.3 - 2021-09-18

- Fixes processing of `mira` files without geolocation attributes

## 1.20.2 - 2021-09-17

- Exceptions module
- Small fixes

## 1.20.1 - 2021-09-16

- Small fixes

## 1.20.0 - 2021-09-14

- Support for Vaisala CL61-D ceilometer

## 1.19.0 - 2021-09-09

- Function to efficiently append data to existing netCDF file
- HATPRO timestamp sorting and time unit fix
- Small bug fixes

## 1.18.3 - 2021-08-26

- Include missing config files

## 1.18.2 - 2021-08-25

- Quality control routines as a part of CloudnetPy installation package
- Small fixes

## 1.18.0 - 2021-08-13

- Improved classification of insects
- 100 m minimum requirement for liquid layers
- Bug fixes

## 1.17.0 - 2021-06-28

- Explicit `_FillValue` attributes
- File format documentation

## 1.16.0 - 2021-06-16

- Filter for stripe-shaped radar artifacts
- Improved error messages and logging
- Small bug fixes

## 1.15.0 - 2021-05-17

- Filtering of bad quality HATPRO profiles
- 1st range gate artifact removal from RPG radar data
- Bug fixes

## 1.14.2 - 2021-05-05

Fixes `ct25k` processing

## 1.14.1 - 2021-04-26

- Store `height` in radar files and use in plots
- Bug fixes

## 1.13.3 - 2021-03-24

- Save calibration factor and site altitude in `lidar` file
- Check for invalid model files
- Bug fixes

## 1.13.0 - 2021-03-18

- Takes ceilometer calibration values as argument to `ceilo2nc`.

## 1.12.0 - 2021-03-16

- Optional time stamp validation for Vaisala ceilometers

## 1.11.0 - 2021-03-14

- Screening of invalid HATPRO time steps
- Plotting improvements
- Bug fixes

## 1.10.2 - 2021-03-10

- Fix `palaiseau` and `lindenberg` chm15k calibration factors
- Improve `mwr` plots

## 1.10.1 - 2021-03-08

- Cloud top and base variables to classification file
- Support for incomplete model files
- Support for concatenating `NETCDF4` formatted files
- Bug fixes, minor method improvements and refactoring

## 1.9.4 - 2021-02-26

- Fixes bug that misplaced RPG cloud radar time array

## 1.9.3 - 2021-02-24

- Bug fixes to HATPRO conversion

## 1.9.2 - 2021-02-19

- Fix classification to work with radars without `LDR` and `width`

## 1.9.1 - 2021-02-18

Fixes a bug that prevented file to be closed properly.

## 1.9.0 - 2021-02-18

This Release:
- Adds support for BASTA cloud radar
- Adds support for HATPRO binary .LWP files
- Fixes `units` of `time` variable
- Fixes several smallish issues and bugs
- Adds loads of typehints and refactoring


## 1.8.2 - 2021-02-05

Better fix for the MIRA timestamp issue

## 1.8.1 - 2021-02-04

Fixes bug that raised error if the last MIRA timestamp was at 24:00.

## 1.8.0 - 2021-02-03

- Library code for concatenating netCDF files
- Option to provide folder name containing `.mmclx` files to `mira2nc` function.

## 1.7.0 - 2020-12-25

- Optional date parameter to mira2nc
- Check that all profiles in a MIRA file have identical date
- Small fixes

## 1.6.1 - 2020-12-17

- Liquid water path plotting bug fix

## 1.6.0 - 2020-12-11

Function for creating images from the legacy files for the data portal

## 1.5.0 - 2020-12-08

- Optional `date` parameter for rpg2nc to validate date in the input files
- MWR plotting for operational processing
- Bug fixes

## 1.4.0 - 2020-11-29

- Optional `uuid` parameter to processing API functions.
- Lindenberg ceilometer calibration value
- Minor fixes

## 1.3.2 - 2020-09-28

This release fixes bug in the RPG timestamp to date conversion.

## 1.3.1 - 2020-09-23

This release adds support for RPG Level 1 V4 files

## 1.3.0 - 2020-09-16

- replace global attribute "source" with "source_file_uuids" for categorize file and level 2 products to enable provenance on the data portal
- add more references to global attribute "references"
- minor fixes

## 1.2.4 - 2020-09-02

This release fixes a bug that causes rpg2nc reader to fail with Python 3.8.

## 1.2.3 - 2020-09-02

This release adds the required modifications and updates from the JOSS review process.

## 1.2.2 - 2020-08-10

This release fixes the bug https://github.com/actris-cloudnet/cloudnetpy/issues/9

## 1.2.1 - 2020-06-03



## 1.2.0 - 2020-06-02

This version adds option to omit title from plotted figures.

## 1.1.0 - 2020-05-11

Version `1.1.0` adds bug fixes and minor changes to high-level API functions:

- Option to keep existing UUID
- UUIDs as return values

## 1.0.7 - 2020-02-13

This is the first CloudnetPy release under actris-cloudnet organization account. The commit history has been truncated. The original repository, which is no longer updated, contains full (and messy) commit history and can be accessed on https://github.com/tukiains/cloudnetpy-legacy.
