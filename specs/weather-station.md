# Cloudnet weather station file format

⚠️ **This document is work in progress!** ⚠️

Send weather station data to Cloudnet in a netCDF file.

## Variables

File should contains variables with the following standard names:

| Standard name                  | Unit   | Description                                                                                                                                                                                       |
| ------------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `time`                         |        | Time of measurement in UTC.                                                                                                                                                                       |
| `wind_speed`                   | m s-1  |                                                                                                                                                                                                   |
| `air_temperature`              | °C     |                                                                                                                                                                                                   |
| `relative_humidity`            | 1      |                                                                                                                                                                                                   |
| `wind_from_direction`          | degree | The direction from which the wind is blowing. The direction increases clockwise such that a northerly wind is 0°, an easterly wind is 90°, a southerly wind is 180°, and a westerly wind is 270°. |
| `air_pressure`                 | Pa     |                                                                                                                                                                                                   |
| `rainfall_rate`                | mm h-1 |                                                                                                                                                                                                   |
| `thickness_of_rainfall_amount` | mm     | Cumulated rainfall since 00:00:00 UTC.                                                                                                                                                            |

## References

[CF Standard Name Table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
