# Changelog

## 1.75.1 – 2025-05-08

- Use Ze as a fallback for old mmclx files

## 1.75.0 – 2025-05-02

- Support LHATPRO and LHUMPRO

## 1.74.4 – 2025-04-25

- Fix mask-ignore warnings

## 1.74.3 – 2025-04-22

- Ignore altitude from GPS

## 1.74.2 – 2025-04-17

- Fix Limassol weather station time

## 1.74.1 – 2025-04-16

- Support Limassol weather station
- Add missing WindCube metadata

## 1.74.0 – 2025-04-15

- Support categorize file with changing location

## 1.73.2 – 2025-04-10

- Fix MWRpy products with moving coordinates

## 1.73.1 – 2025-04-10

- Write latitude and longitude as vector if they are not stationary

## 1.73.0 – 2025-04-07

- Add bowtie2nc for processing RV Meteor radar data (#115)

## 1.72.3 – 2025-04-04

- Fix HATPRO coordinate handling

## 1.72.2 – 2025-04-03

- Skip zero GPS coordinates

## 1.72.1 – 2025-04-03

- Fix reading RPG radar coordinates

## 1.72.0 – 2025-04-03

- Prefer instrument's GPS coordinates over site coordinates
- Screen invalid temperature and pressure from RPG cloud radar

## 1.71.6 – 2025-03-31

- Fix bug related to time attribute in MWRpy

## 1.71.5 – 2025-03-31

- Fix IR dimension name in MWRpy
- Support altitude vectors in plotting

## 1.71.4 – 2025-03-24

- Read ceilometer ASCII files using [`ceilopyter`](https://pypi.org/project/ceilopyter/) package

## 1.71.3 – 2025-03-14

- Extend CS135 support
- Fix frequency formatting in ier and lwc comments

## 1.71.2 – 2025-03-07

- Handle all data masked in epsilon-lidar

## 1.71.1 – 2025-03-06

- Fix epsilon-lidar metadata

## 1.71.0 – 2025-03-05

- Add generate epsilon-lidar product

## 1.70.2 – 2025-03-04

- Check if all BASTA data is masked
- Fix units formatting in plots

## 1.70.1 – 2025-02-13

- Add plotting metadata for EarthCARE velocity variables

## 1.70.0 – 2025-02-11

- Add cpr-simulation plotting

## 1.69.10 – 2025-02-10

- Screen invalid rainfall rates in Thies LPM

## 1.69.9 – 2025-02-10

- Skip null characters Lampedusa weather station
- Remove `WeatherStationDataError`

## 1.69.8 – 2025-02-10

- Add support for Lampedusa weather station

## 1.69.7 – 2025-02-07

- Support `time_offset` in `hatpro2l1c`
- Test with Python 3.13

## 1.69.6 – 2025-02-05

- Ignore masked mira timestamps

## 1.69.5 – 2025-02-04

- Fix bug in calibration data reading

## 1.69.4 – 2025-02-03

- Add snr_limit metadata option to mira
- Store snr_limit variable in netCDF

## 1.69.3 – 2025-01-31

- Improve `chm15k` version check

## 1.69.2 – 2025-01-29

- Handle invalid timestamps in CT25K

## 1.69.1 – 2025-01-29

- Support zenith and azimuth offsets in MIRA

## 1.69.0 – 2025-01-29

- Support Jülich weather station
- Add ARPEGE to supported models

## 1.68.1 – 2025-01-17

- Improve rain[e]H3 timestamp screening

## 1.68.0 – 2025-01-17

- Add rain[e]H3 rain gauge processing (#113)

## 1.67.8 – 2025-01-16

- Handle old MIRA files without angle variables

## 1.67.7 – 2025-01-14

- Add rain gauge metadata

## 1.67.6 – 2025-01-07

- Fix masking of zero values in RPG data

## 1.67.5 – 2024-12-18

- Fix MWRpy metadata
- Skip completely masked PollyXT profiles
- Skip files raising HDF error in concatenation

## 1.67.4 – 2024-12-09

- Check for empty Copernicus and Galileo data
- Move magic numbers into arguments

## 1.67.3 – 2024-12-05

- Fix memory error with Matplotlib
- Improve PollyXT data reading

## 1.67.2 – 2024-12-03

- Remove default SNR limit

## 1.67.1 – 2024-12-03

- Adjust MIRA-10 screening

## 1.67.0 – 2024-12-03

- Add MIRA-10 support
- Add `plot_above_ground` option
- Fix height calculating with masked zenith angle

## 1.66.19 – 2024-12-02

- Add option for minor ticks
- Add doppler-lidar plot max gap
- Add windcube models

## 1.66.18 – 2024-11-27

- Improve MIRA angle screening

## 1.66.17 – 2024-11-22

- Assume negative MIRA elevation angles are vertical measurements

## 1.66.16 – 2024-11-18

- Allow small differences between files with PollyXT data

## 1.66.15 – 2024-11-18

- Improve MIRA zenith angle screening
- Avoid single profile categorize file
- Handle out-of-order data from Radiometrics
- Skip file with missing variables in concatenation

## 1.66.14 – 2024-11-18

- Handle different number of measurements from Radiometrics

## 1.66.13 – 2024-11-15

- Read profiles from Radiometrics files

## 1.66.12 – 2024-11-14

- Fix `VOODOO` global attributes

## 1.66.11 – 2024-11-14

- Remove snowfall from categorize rain detection

## 1.66.10 – 2024-11-01

- Update reader for Galati weather station

## 1.66.9 – 2024-10-31

- Fix check for empty index array

## 1.66.8 – 2024-10-30

- Raise exception if all timestamps are screened
- Fix CLI product parsing

## 1.66.7 – 2024-10-22

- Fix cli product parse

## 1.66.6 – 2024-10-22

- Optimize `rebin_2d`
- Add `raise_on_empty` option
- Add CloudnetPy CLI

## 1.66.5 – 2024-10-15

- Check for missing time data
- Improve plotting of zero temperature line

## 1.66.4 – 2024-10-07

- Support `ecmwf-open` model

## 1.66.3 – 2024-10-07

- Improve MIRA data screening

## 1.66.2 – 2024-10-03

- Support variation of Thies LNM format from pyAtmosLogger

## 1.66.1 – 2024-09-30

- Use flags in moving average plot only with MWRpy products

## 1.66.0 – 2024-09-24

- Implement rain and melting layer attenuations
- Calculate liquid and gas attenuation using ITU recommendations
- Calculate wet-bulb temperature using iterative method
- Fix rain detection when data is masked
- Plot zero line for wet-bulb temperature
- Don't flag IRT data in MWR plot
- Change "Wavel" to "WL" in MWR plot
- Sort values in `source_file_uuids` attribute

## 1.65.8 – 2024-09-06

- Fix wind direction plotting

## 1.65.7 – 2024-09-05

- Fix bug in liquid detection if all LWP are masked

## 1.65.6 – 2024-09-05

- Fix `mwr-l1c` plotting

## 1.65.5 – 2024-09-05

- Improve moving average calculation in 1D plots
- Include error message from MWRpy

## 1.65.4 – 2024-08-30

- Fix weather station variable names and units of RPG cloud radars

## 1.65.3 – 2024-08-27

- Support pyAtmosLogger format for Thies LNM

## 1.65.2 – 2024-08-21

- Check if all data are masked in mwr-single and mwr-multi

## 1.65.1 – 2024-08-21

- Add check for valid brightness temperatures

## 1.65.0 – 2024-08-20

- Make geophysical products work without LWP where possible

## 1.64.5 – 2024-08-07

- Use `lwp_quality_flag` if exists
- Avoid divide by zero warning

## 1.64.4 – 2024-08-06

- Adjust screening of old chm15k data
- Skip smoothing-based filtering when data are range-uncorrected

## 1.64.3 – 2024-08-05

- Fix bug in altitude above ground calculation

## 1.64.2 – 2024-08-02

- Increase allowed gap for `mwr-multi`

## 1.64.1 – 2024-08-02

- Sort hatpro timestamps after conversion to fraction hour

## 1.64.0 – 2024-08-01

- Add `temperature_offset` option to wet bulb temperature

## 1.63.0 – 2024-08-01

- Improve method for detecting rain from radar
- Increase particle limit in disdrometer rain detection
- Remove rain smoothing when screening insects

## 1.62.1 – 2024-07-31

- Support missing values in Granada weather station

## 1.62.0 – 2024-07-31

- Convert lidar liquid colder than -38 C into ice
- Use cl61d noise screening with more ceilometers

## 1.61.17 – 2024-07-30

- Add support for Galati and Bucharest weather stations
- Fix calculation of average wind direction

## 1.61.16 – 2024-07-25

- Fix `seconds2date` with NumPy 2.0

## 1.61.15 – 2024-06-12

- Improve CITATION.cff

## 1.61.14 – 2024-06-05

- Fix hyytiala weather-station invalid data screening

## 1.61.13 – 2024-06-05

- Fix legacy file plotting
- Add support for hyytiala weather-station

## 1.61.12 – 2024-05-29

- Change plotting of models

## 1.61.11 – 2024-05-23

- Fix `rainfall_amount` calculation
- Fix `source` global attribute in weather station

## 1.61.10 – 2024-05-22

- Fix Kenttärova rainfall rate units

## 1.61.9 – 2024-05-22

- Support Kenttärova weather station

## 1.61.8 – 2024-05-15

- Handle incorrect data length in Vaisala ceilometer
- Harmonize `definition` attribute formatting

## 1.61.7 – 2024-05-03

- Skip Copernicus and Galileo files with missing variables

## 1.61.6 – 2024-04-29

- Support Granada weather station
- Add `instrument_pid` attribute to `mwr-single` and `mwr-multi`

## 1.61.5 – 2024-04-18

- Fix missing `n_particles` in categorize generation

## 1.61.4 – 2024-04-18

- Support custom truncated Thies LNM format

## 1.61.3 – 2024-04-17

- Fix Thies LNM reader

## 1.61.2 – 2024-04-17

- Handle unreadable Thies LNM file gracefully

## 1.61.1 – 2024-04-17

- Fix truncation in unit conversion

## 1.61.0 – 2024-04-17

- Support TOA5 format for Thies LNM
- Allow truncation in other than time dimension

## 1.60.4 – 2024-04-05

- Fix `clabel` fallback when using custom `plot_meta`

## 1.60.3 – 2024-03-21

- Remove NaNs from L3 files
- Return dimensions from L3 plotting

## 1.60.2 – 2024-03-11

- Fix L3 plot issues

## 1.60.1 – 2024-03-06

- Improve stripe filter
- Fix Radiometrics units

## 1.60.0 – 2024-02-27

- Screen completely masked profiles from PollyXT data
- Use double precision in PollyXT time vector

## 1.59.5 – 2024-02-25

- Remove zero-line from legend

## 1.59.4 – 2024-02-25

- Fix 2d smoothing to work with masked profiles

## 1.59.3 – 2024-02-23

- Add zero-line to lwp plot
- Add smoothing to 2d mwrpy plots

## 1.59.2 – 2024-02-22

- Add windcube instrument

## 1.59.1 – 2024-02-21

- Add reader for Bucharest Parsivel

## 1.59.0 – 2024-02-15

- Add doppler-lidar-wind product

## 1.58.8 – 2024-02-14

- Check measurement date in MWRpy-processed data
- Use double precision for BASTA time
- Handle truncated Vaisala ceilometer file
- Skip invalid lines in Thies LNM file

## 1.58.7 – 2024-02-08

- Include both elevation and zenith angle in MWRpy products
- Test with Python 3.12

## 1.58.6 – 2024-02-06

- Skip NaNs when plotting `mwr-l1c`

## 1.58.5 – 2024-02-06

- Fix plotting of brightness temperatures
- Fix dimensions of `ir_wavelength` variable
- Show wavelength in IR plots

## 1.58.4 – 2024-02-05

- Support plotting infrared brightness temperature
- Disable voodoonet progress bar

## 1.58.3 – 2024-02-01

- Make sure only RPG cloud radar is used with Voodoo

## 1.58.2 – 2024-01-31

- Adjust variable metadata
- Allow 12 min gap in weather station plots

## 1.58.1 – 2024-01-18

- Support missing values in Parsivel reader

## 1.58.0 – 2024-01-18

- Fix Z_error calculation to follow Hogan 1998
- Fix error for radars without width
- Implement disdrometer data in categorize file

## 1.57.0 – 2024-01-17

- Support plotting of additional Parsivel variables
- Mask invalid values in `number_concentration` and `fall_velocity`
- Fix units of `number_concentration` variable
- Share x-axis on multiple plots

## 1.56.14 – 2024-01-15

- Support yet another Parsivel format
- Fix line break in a netCDF definition

## 1.56.13 – 2024-01-09

- Fix moving average plotting when no gaps
- Check parsivel time vector

## 1.56.12 – 2024-01-08

- Fix unit in rainfall rate plots

## 1.56.11 – 2023-12-16

- Correct voodoo liquid tops

## 1.56.10 – 2023-12-15

- Remove radar-detected liquid at warm temperatures

## 1.56.9 – 2023-12-13

- Fix colorbar label of certain 2d plots
- Show frequency in brightness temperature plot

## 1.56.8 – 2023-12-12

- Adjust plotting
- Fix a few small bugs and other issues

## 1.56.7 – 2023-12-08

- Add dimension check to moving average plot

## 1.56.6 – 2023-12-08

- Fix moving average plot in presence of flagged data
- Adjust flagged region style
- Add PlottingError exception

## 1.56.5 – 2023-12-05

- Improve mwrpy processing and plotting (#97)

## 1.56.4 – 2023-12-04

- Fix "/" separator in Parsivel spectra

## 1.56.3 – 2023-12-01

- Improve quickstart documentation

## 1.56.2 – 2023-12-01

- Host documentation on GitHub Pages

## 1.56.1 – 2023-11-30

- Avoid crashing if source attribute is missing
- Make plotting work with legacy files

## 1.56.0 – 2023-11-30

- `generate_figure` now takes plotting options as single argument

## 1.55.23 – 2023-11-23

- Add CloudnetException

## 1.55.22 – 2023-11-23

- Do not mask zeros from integer data types and rainfall_rate
- Use milliseconds in RPG time
- Fix bug in time index order

## 1.55.21 – 2023-11-20

- Remove masked model profiles from categorize

## 1.55.20 – 2023-11-08

- Add yet another Parsivel reader

## 1.55.19 – 2023-11-08

- Fix mwr-l1c processing without IR data

## 1.55.18 – 2023-10-23

- Raise error if 'height' missing from model file

## 1.55.17 – 2023-10-23

- Remove duplicate timestamps from BASTA
- Raise error if all MIRA data are masked

## 1.55.16 – 2023-10-19

- Support MIRA files with NyquistVelocity as array
- Add a few standard names

## 1.55.15 – 2023-10-11

- Fix metadata of L3 products

## 1.55.14 – 2023-10-08

- Avoid abrupt transitions between aerosol and ice

## 1.55.13 – 2023-10-08

- Plot radar `lwp`
- Check `beta_raw` shape

## 1.55.12 – 2023-10-06

- Use `beta_smooth` mask for beta with CL61

## 1.55.11 – 2023-10-06

- Screen nan values from Vaisala ceilometers
- Store raw CL61 variables

## 1.55.10 – 2023-10-04

- Skip Vaisala ceilometer lines with bad data

## 1.55.9 – 2023-09-29

- Fix calibration offset unit
- Fix Copernicus range issues

## 1.55.8 – 2023-09-28

- Screen MIRA ldr in STSR mode
- Mask Copernicus data below 150m
- Detect if all MWR data are low quality

## 1.55.7 – 2023-09-27

- Improve RPG-FMCW-94 zenith angle screening

## 1.55.6 – 2023-09-26

- Filter RPG-FMCW-94 containing scan data

## 1.55.5 – 2023-09-26

- Do not process corrupted RPG-FMCW-94 files
- Allow some RPG cloud radar header values to vary

## 1.55.4 – 2023-09-25

- Mask negative PollyXT beta

## 1.55.3 – 2023-09-21

- Improve Copernicus noise screening
- Fix disdrometer data_raw data type

## 1.55.2 – 2023-09-20

- Adjust parameters of CT25K background screening
- Update weather station file format specification
- Fix many warnings

## 1.55.1 – 2023-09-18

- Improve global attributes of `mwrpy` products

## 1.55.0 – 2023-09-15

- Include serial number for Thies LNM
- Add more options to `generate_figure`
- Add `timestamps` argument to `parsivel2nc`

## 1.54.1 – 2023-09-13

- Skip invalid MRR-PRO input files

## 1.54.0 – 2023-09-13

- Add initial support for MRR-PRO
- Fix the bug in the code that distinguishes cl31 from cl51

## 1.53.2 – 2023-09-08

- Fix crash on HATPRO files with one profile

## 1.53.1 – 2023-09-06

- Speed up processing of .znc files by dropping spectra (#87)
- Remove network call from model-related code

## 1.53.0 – 2023-08-28

- Support `.znc` input (and STSR) in `mira2nc` (#84)

## 1.52.3 – 2023-08-25

- Add VoodooNet version to output file

## 1.52.2 – 2023-08-21

- Add `source_file_uuids` attribute to `mwr_single` and `mwr_multi`

## 1.52.1 – 2023-08-15

- Catch mwrpy exception

## 1.52.0 – 2023-08-11

- Fix brightness temperature plotting
- Use revised mwrpy functions

## 1.51.1 – 2023-08-01

- Support CL61d and PollyXT serial numbers
- Fix CL61d zenith angle
- Improve CS135 reader

## 1.51.0 – 2023-07-26

- Support Parsivel from Campbell Scientific CR1000 datalogger
- Fix subtitle of the second plot in `compare_files` (#82)

## 1.50.0 – 2023-06-28

- Add support for `cs135`

## 1.49.9 – 2023-06-27

- Add option to give ceilometer model to ceilo2nc
- Update fileformat.rst

## 1.49.8 – 2023-06-12

- Fix reading of truncated lines in Parsivel

## 1.49.7 – 2023-06-09

- Support input files as list in mira2nc (#81)

## 1.49.6 – 2023-06-01

- Improve duplicate timestamp handling in HATPRO reader

## 1.49.5 – 2023-05-16

- Extend mwrpy processing to all sites

## 1.49.4 – 2023-05-12

- Fix `mwr-multi` time units

## 1.49.3 – 2023-05-12

- Re-release due to PyPI failure

## 1.49.2 – 2023-05-12

- Add potential temperature plot meta
- Improve global attributes

## 1.49.1 – 2023-05-11

- Fix lwp plotting

## 1.49.0 – 2023-05-11

- Implement mwrpy products
- Support more values in Parsivel telegram

## 1.48.0 – 2023-05-10

- Change `lwp` unit to `kg m-2`

## 1.47.2 – 2023-05-08

- Write CHM15k serial number to output file

## 1.47.1 – 2023-05-05

- Allow unknown values is Parsivel telegram

## 1.47.0 – 2023-05-05

- Mask unrealistic `der` values
- Split `disdrometer2nc` into `parsivel2nc` and `thies2nc`
- Support more Parsivel format variants
- Support Python 3.11

## 1.46.5 – 2023-04-18

- Make Radiometrics reader more flexible
- Read IWV from Radiometrics

## 1.46.4 – 2023-03-24

- Accept nonzero but constant `azimuth_velocity`
- Handle masked `zenith` and `azimuth` values

## 1.46.3 – 2023-03-19

- Improve `mira` global attribute parsing
- Allow inconsistent `ovl` in `mira` data

## 1.46.2 – 2023-03-15

- Add `galileo` clutter screening

## 1.46.1 – 2023-03-14

- Plot wind direction with dots

## 1.46.0 – 2023-03-10

- Add `galileo` cloud radar processing

## 1.45.1 – 2023-03-01

- Add fallback for `rainfall_rate`
- Mask `nan` values in weather station data
- Adjust `rainfall_amount` plot

## 1.45.0 – 2023-03-01

- Add weather station processing
- Replace `rain_rate` with `rainfall_rate`
- Harmonize metadata definitions
- Migrate model-evaluation documentation

## 1.44.2 – 2023-02-07

- Use more specific exceptions in PollyXT handling

## 1.44.1 – 2023-01-13

- Update `cloudnetpy-qc` version requirement
- Add rv-polarstern pollyxt variables
- Update LICENSE
- Add `atmos_utils.py` to get rid of cyclic import
- Use human-readable `pylint` problem names

## 1.44.0 – 2022-12-21

- Fix mask in scaled `der` variables
- Change `ier` unit to m
- Write `liquid_prob` to categorize file
- Add references to `categorize`, `ier` and `der` files

## 1.43.1 – 2022-12-15

- Avoid crashing when different number of hatpro `.LWP` and `.IWV` files

## 1.43.0 – 2022-12-13

- Use `voodoonet` for improving liquid detection
- Simplify `find_liquid` function
- Fix bug in screening function with 3d data
- Fix disdrometer metadata
- Use Python3.10 features
- Update Python requirement to 3.10
- Remove pytz

## 1.42.2 – 2022-11-23

- Improve disdrometer product writing

## 1.42.1 – 2022-11-23

- Check for empty time vector

## 1.42.0 – 2022-11-22

- Use `sldr` for insect detection
- Mask invalid parsivel data values (#68)
- Fix bug causing IndexError in melting layer detection

## 1.41.2 – 2022-11-20

- Allow scalar variable `nave` values to change between concatenated `mira` files

## 1.41.1 – 2022-11-18

- Fix parsivel `number_concentration` units to pass cfchecks

## 1.41.0 – 2022-11-18

- Migrate model-evaluation to cloudnetpy

## 1.40.0 – 2022-11-17

- Raise ValidTimeStampError from disdrometer processing
- Remove duplicate timestamps from disdrometer data
- Adjust potential melting layer temperature range calculation
- Deprecate general.py module and RadarArray class

## 1.39.0 – 2022-10-18

- Add option to ignore variables from the concatenation
- Fix to work with older BASTA files

## 1.38.0 – 2022-10-16

- Avoid classifying lidar-only signals as ice close to surface

## 1.37.1 – 2022-10-14

- Allow sample_duration to vary

## 1.37.0 – 2022-10-12

- Use `rpgpy` to read `rpg-fmcw-94` files

## 1.36.4 – 2022-10-07

- Check that files to concatenate have same values in variables
- Write SLDR `long_name` attribute to categorize file
- Test with Python 3.10

## 1.36.3 – 2022-09-14

- Fix error when no overlapping timestamps in categorize

## 1.36.2 – 2022-08-23

- Support inconsistent time vector in HATPRO files
- Improve HATPRO and PollyXT error handling
- Add standard name for IWV

## 1.36.1 – 2022-08-18

- Adjust copernicus outlier screening

## 1.36.0 – 2022-08-17

- Change `solar_azimuth_angle` to `sensor_azimuth_angle`
- Add support for Copernicus cloud radar
- Optimize HATPRO binary file reading speed

## 1.35.0 – 2022-08-11

- Support HATPRO \*.IWV files
- Adjust plotting parameters of RPG radar
- Restore insect probability

## 1.34.0 – 2022-06-20

- Use fallback pollyXT backscatter channel

## 1.33.2 – 2022-06-15

- Add sorting of `basta` timesteps

## 1.33.1 – 2022-06-06

- Add solid and total rainfall rate (#53)

## 1.33.0 – 2022-05-24

- Add ice effective radius product (#51)
- Add droplet effective radius product (#50)
- Run tests on pull request

## 1.32.0 – 2022-05-13

- Skip corrupted profiles in CL51 files
- Add missing units
- Test Windows and macOS
- Test Python 3.10 support
- Fix code formatting

## 1.31.2 – 2022-03-23

- Cast `parsivel` metadata to floats

## 1.31.1 – 2022-03-22

- Allow `rpg-fmcw-94` latitude and longitude values to vary
- Raise error if only one valid `mira` timestamp

## 1.31.0 – 2022-03-14

- Drizzle without spectral width
- Minor fixes

## 1.30.2 – 2022-03-07

- Bug fix

## 1.30.1 – 2022-03-04

- Bug fixes

## 1.30.0 – 2022-03-04

- Adds support for `chm15k` ceilometers that contain `beta_att` instead of `beta_raw`
- Separates test-dependencies in `setup.py`
- Fixes myriad type hints

## 1.29.4 – 2022-02-15

`rpg-fmcw-94` fixes:

- Does not crash if elevation angle missing
- Sorts timestamps and removes duplicates

## 1.29.3 – 2022-02-09

- Adds screening of small `rpg-fmcw-94` width values.

## 1.29.2 – 2022-02-07

- Fix chm15k(x) source attribute

## 1.29.1 – 2022-02-02

- Add HALO Doppler lidar to `instruments.py`

## 1.29.0 – 2022-01-31

- Add Radiometrics support

## 1.28.1 – 2022-01-24

- Use the same plotting routines for current and legacy files

## 1.28.0 – 2022-01-17

- Return dimensions of generated images

## 1.27.7 – 2021-12-21

- Adds timestamp sorting and duplicate removal for `mira`

## 1.27.6 – 2021-12-21

- Raises custom exception from bad model file

## 1.27.5 – 2021-12-20

- Removes duplicate `hatpro` timestamps

## 1.27.4 – 2021-12-19

- Sort and remove duplicates from Vaisala cl31/51 data

## 1.27.3 – 2021-12-16

- Correctly classify first layer as ice

## 1.27.2 – 2021-12-16

- Fixes bug in freezing region determination

## 1.27.1 – 2021-12-15

- Fix bug in model plotting

## 1.27.0 – 2021-12-15

- Lidar / radar data gaps removed from the time array
- Data gaps shown as vertical grey bars in classification / status plots
- Level 2 products harmonized against legacy files and netCDF validator

## 1.26.0 – 2021-12-09

- Improved melting layer detection
- Improved drizzle / insects classification
- Detection status variable to match legacy files
- Minor fixes to plotting and classification file metadata

## 1.25.1 – 2021-11-29

- Removes quality control from CloudnetPy package
- Adds speckle filter to BASTA data
- Removes classification results from profiles without any lidar data

## 1.25.0 – 2021-11-29

- Uses Python 3.8 and newer. Older Python versions are not supported from now on.
- Updates categorize file:
  - Fixes netCDF metadata
  - Interpolates lidar data using nearest neighbor
  - Adds 1px melting layer when not detected from data

## 1.24.0 – 2021-11-22

- Harmonized `hatpro` processing

## 1.23.2 – 2021-11-18

- Screening of `mira` files with deviating height vector
- Screening of `mira` profiles with deviating zenith angle
- snr limit option to `pollyxt2nc`

## 1.23.0 – 2021-11-16

- Refactored radar processing
- `instruments.py` module
- Minor fixes to global attributes
- Improved tests
- Deprecated `keep_uuid` option

## 1.22.4 – 2021-11-09

- Fix processing of old `chm15k` ceilometers

## 1.22.3 – 2021-11-08

- Minor tuning of the ceilometer screening method

## 1.22.1 – 2021-11-05

- Improved ceilometer data screening

## 1.22.0 – 2021-10-30

- PollyXT support
- Harmonized lidar files

## 1.21.2 – 2021-10-09

- Avoid crash in `update_nc` if invalid nc file
- QC adjustment

## 1.21.1 – 2021-09-29

- Custom exception for disdrometer files that can not be read

## 1.21.0 – 2021-09-28

- Initial support for `Parsivel2` and `Thies-LNM` disdrometers
- Quality control adjustments

## 1.20.4 – 2021-09-21

- Adds more contrast to ice clouds

## 1.20.3 – 2021-09-18

- Fixes processing of `mira` files without geolocation attributes

## 1.20.2 – 2021-09-17

- Exceptions module
- Small fixes

## 1.20.1 – 2021-09-16

- Small fixes

## 1.20.0 – 2021-09-14

- Support for Vaisala CL61-D ceilometer

## 1.19.0 – 2021-09-09

- Function to efficiently append data to existing netCDF file
- HATPRO timestamp sorting and time unit fix
- Small bug fixes

## 1.18.3 – 2021-08-26

- Include missing config files

## 1.18.2 – 2021-08-25

- Quality control routines as a part of CloudnetPy installation package
- Small fixes

## 1.18.0 – 2021-08-13

- Improved classification of insects
- 100 m minimum requirement for liquid layers
- Bug fixes

## 1.17.0 – 2021-06-28

- Explicit `_FillValue` attributes
- File format documentation

## 1.16.0 – 2021-06-16

- Filter for stripe-shaped radar artifacts
- Improved error messages and logging
- Small bug fixes

## 1.15.0 – 2021-05-17

- Filtering of bad quality HATPRO profiles
- 1st range gate artifact removal from RPG radar data
- Bug fixes

## 1.14.2 – 2021-05-05

Fixes `ct25k` processing

## 1.14.1 – 2021-04-26

- Store `height` in radar files and use in plots
- Bug fixes

## 1.13.3 – 2021-03-24

- Save calibration factor and site altitude in `lidar` file
- Check for invalid model files
- Bug fixes

## 1.13.0 – 2021-03-18

- Takes ceilometer calibration values as argument to `ceilo2nc`.

## 1.12.0 – 2021-03-16

- Optional time stamp validation for Vaisala ceilometers

## 1.11.0 – 2021-03-14

- Screening of invalid HATPRO time steps
- Plotting improvements
- Bug fixes

## 1.10.2 – 2021-03-10

- Fix `palaiseau` and `lindenberg` chm15k calibration factors
- Improve `mwr` plots

## 1.10.1 – 2021-03-08

- Cloud top and base variables to classification file
- Support for incomplete model files
- Support for concatenating `NETCDF4` formatted files
- Bug fixes, minor method improvements and refactoring

## 1.9.4 – 2021-02-26

- Fixes bug that misplaced RPG cloud radar time array

## 1.9.3 – 2021-02-24

- Bug fixes to HATPRO conversion

## 1.9.2 – 2021-02-19

- Fix classification to work with radars without `LDR` and `width`

## 1.9.1 – 2021-02-18

- Fixes a bug that prevented file to be closed properly.

## 1.9.0 – 2021-02-18

This Release:

- Adds support for BASTA cloud radar
- Adds support for HATPRO binary .LWP files
- Fixes `units` of `time` variable
- Fixes several smallish issues and bugs
- Adds loads of typehints and refactoring

## 1.8.2 – 2021-02-05

- Better fix for the MIRA timestamp issue

## 1.8.1 – 2021-02-04

Fixes bug that raised error if the last MIRA timestamp was at 24:00.

## 1.8.0 – 2021-02-03

- Library code for concatenating netCDF files
- Option to provide folder name containing `.mmclx` files to `mira2nc` function.

## 1.7.0 – 2020-12-25

- Optional date parameter to mira2nc
- Check that all profiles in a MIRA file have identical date
- Small fixes

## 1.6.1 – 2020-12-17

- Liquid water path plotting bug fix

## 1.6.0 – 2020-12-11

Function for creating images from the legacy files for the data portal

## 1.5.0 – 2020-12-08

- Optional `date` parameter for rpg2nc to validate date in the input files
- MWR plotting for operational processing
- Bug fixes

## 1.4.0 – 2020-11-29

- Optional `uuid` parameter to processing API functions.
- Lindenberg ceilometer calibration value
- Minor fixes

## 1.3.2 – 2020-09-28

This release fixes bug in the RPG timestamp to date conversion.

## 1.3.1 – 2020-09-23

This release adds support for RPG Level 1 V4 files

## 1.3.0 – 2020-09-16

- replace global attribute "source" with "source_file_uuids" for categorize file and level 2 products to enable provenance on the data portal
- add more references to global attribute "references"
- minor fixes

## 1.2.4 – 2020-09-02

- This release fixes a bug that causes rpg2nc reader to fail with Python 3.8.

## 1.2.3 – 2020-09-02

- This release adds the required modifications and updates from the JOSS review process.

## 1.2.2 – 2020-08-10

- This release fixes the bug https://github.com/actris-cloudnet/cloudnetpy/issues/9

## 1.2.1 – 2020-06-03

## 1.2.0 – 2020-06-02

- This version adds option to omit title from plotted figures.

## 1.1.0 – 2020-05-11

Version `1.1.0` adds bug fixes and minor changes to high-level API functions:

- Option to keep existing UUID
- UUIDs as return values

## 1.0.7 – 2020-02-13

This is the first CloudnetPy release under actris-cloudnet organization account. The commit history has been truncated. The original repository, which is no longer updated, contains full (and messy) commit history and can be accessed on https://github.com/tukiains/cloudnetpy-legacy.
