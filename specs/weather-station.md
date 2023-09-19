# Cloudnet weather station file format

⚠️ **This document is work in progress!** ⚠️

File format is comma-separated values (CSV).
First line MUST be a header which specify columns used in the file.
Supported columns are documented below.

Valid values MUST be either integers or real numbers.
Decimal separator MUST be dot (e.g. `1.22`).
Missing or invalid values SHOULD be represented with `NaN` and placeholder values like `-99` or `-99.99` SHOULD NOT be used.

## Columns

| Columns                        | Obligation | Unit   | Description                                                                                                                                                                                       |
| ------------------------------ | ---------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `year`                         | mandatory  | 1–9999 | Year of measurement in UTC.                                                                                                                                                                       |
| `month`                        | mandatory  | 1–12   | Month of measurement in UTC.                                                                                                                                                                      |
| `day`                          | mandatory  | 1–31   | Day of measurement in UTC.                                                                                                                                                                        |
| `hour`                         | mandatory  | 0–23   | Hour of measurement in UTC.                                                                                                                                                                       |
| `minute`                       | mandatory  | 0–59   | Minute of measurement in UTC.                                                                                                                                                                     |
| `second`                       | mandatory  | 0–59   | Second of measurement in UTC.                                                                                                                                                                     |
| `wind_speed`                   | optional   | m s-1  |                                                                                                                                                                                                   |
| `air_temperature`              | optional   | C      |                                                                                                                                                                                                   |
| `relative_humidity`            | optional   | 1      |                                                                                                                                                                                                   |
| `wind_from_direction`          | optional   | degree | The direction from which the wind is blowing. The direction increases clockwise such that a northerly wind is 0°, an easterly wind is 90°, a southerly wind is 180°, and a westerly wind is 270°. |
| `air_pressure`                 | optional   | Pa     |                                                                                                                                                                                                   |
| `rainfall_rate`                | optional   | mm h-1 |                                                                                                                                                                                                   |
| `thickness_of_rainfall_amount` | optional   | mm     | Cumulated rainfall since 00:00:00 UTC.                                                                                                                                                            |

Column name MUST be followed by measurement height in parentheses in meters above ground level, for example `wind_speed(2m)`.
It's possible to have same measurement at different heights, for example: `wind_speed(2m)`, `wind_speed(10m)` and `wind_speed(20m)`.

Data CAN be split into several weather-station files. For example, rain-related variables can be in one file and the rest of the variables in other file.
All files must be linked with an instrument PID, unambiguously describing the instrument(s) used ([see example](https://hdl.handle.net/21.12132/3.80082867c5744f11)).

Lines starting with the character `#` are treated as comments and ignored by the parser.

## Examples

Weather station at ground level with various measurements:

```csv
year,month,day,hour,minute,second,wind_speed(10m),wind_from_direction(10m),air_temperature(2m),relative_humidity(2m),air_pressure(2m),rainfall_rate(2m),thickness_of_rainfall_amount(2m)
2019,4,11,0,0,0,1.22,32.75,19.11,76.93,999.40,0.00,0.00
2019,4,11,0,1,0,0.81,321.58,19.06,77.08,999.41,0.00,0.00
2019,4,11,0,2,0,0.52,324.31,19.03,77.17,999.52,0.00,0.00
```

Missing temperature measurements:

```csv
year,month,day,hour,minute,second,wind_speed(10m),wind_from_direction(10m),air_temperature(2m),relative_humidity(2m),air_pressure(2m),rainfall_rate(2m),thickness_of_rainfall_amount(2m)
2019,4,11,0,0,0,1.22,32.75,NaN,76.93,999.40,0.00,0.00
2019,4,11,0,1,0,0.81,321.58,19.06,77.08,999.41,0.00,0.00
2019,4,11,0,2,0,0.52,324.31,NaN,77.17,999.52,0.00,0.00
```

Wind speed measurements at three heights:

```csv
year,month,day,hour,minute,second,wind_speed(2m),wind_speed(10m),wind_speed(20m)
2019,4,11,0,0,0,1.22,1.21,1.24
2019,4,11,0,1,0,0.81,0.83,0.87
2019,4,11,0,2,0,0.52,0.51,0.50
```

File with comments:

```csv
# This is a comment
# This is another comment
year,month,day,hour,minute,second,wind_speed(2m),wind_speed(10m),wind_speed(20m)
2019,4,11,0,0,0,1.22,1.21,1.24
2019,4,11,0,1,0,0.81,0.83,0.87
2019,4,11,0,2,0,0.52,0.51,0.50
```

## References

[CF Standard Name Table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
