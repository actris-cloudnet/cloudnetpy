# Cloudnet weather station file format

⚠️ **This document is work in progress!** ⚠️

File format is comma-separated values (CSV).
First line MUST be a header which specify columns used in the file.
Supported columns are documented below.

Values MUST be either integers or real numbers.
Decimal separator MUST be dot (e.g. `1.22`).
Missing or invalid values SHOULD be left empty and placeholder values, such as `-99.99`, `NaN` or `NA`, SHOULD NOT be used.
For example, row `1,,3` has missing value in second column.

## Columns

| Column                         | Obligation | Unit   | Description                                                                                                                                                                                       |
| ------------------------------ | ---------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `year`                         | mandatory  | 1–9999 | Year of measurement in UTC.                                                                                                                                                                       |
| `month`                        | mandatory  | 1–12   | Month of measurement in UTC.                                                                                                                                                                      |
| `day`                          | mandatory  | 1–31   | Day of measurement in UTC.                                                                                                                                                                        |
| `hour`                         | mandatory  | 0–23   | Hour of measurement in UTC.                                                                                                                                                                       |
| `minute`                       | mandatory  | 0–59   | Minute of measurement in UTC.                                                                                                                                                                     |
| `second`                       | mandatory  | 0–59   | Second of measurement in UTC.                                                                                                                                                                     |
| `wind_speed`                   | optional   | m s-1  |                                                                                                                                                                                                   |
| `air_temperature`              | optional   | K      |                                                                                                                                                                                                   |
| `relative_humidity`            | optional   | 1      |                                                                                                                                                                                                   |
| `wind_from_direction`          | optional   | degree | The direction from which the wind is blowing. The direction increases clockwise such that a northerly wind is 0°, an easterly wind is 90°, a southerly wind is 180°, and a westerly wind is 270°. |
| `air_pressure`                 | optional   | hPa    |                                                                                                                                                                                                   |
| `rainfall_rate`                | optional   | m s-1  |                                                                                                                                                                                                   |
| `thickness_of_rainfall_amount` | optional   | mm     | Cumulated rainfall since 00:00:00 UTC.                                                                                                                                                            |

Multiple measurements at different heights are indicated with multiple columns in format: variable name followed by meters above ground in parentheses.
For instance, wind speed measurements at three different heights would have columns `wind_speed(2m)`, `wind_speed(10m)` and `wind_speed(20m)`.

## Examples

Weather station at ground level with various measurements:

```csv
year,month,day,hour,minute,second,wind_speed,wind_from_direction,air_temperature,relative_humidity,air_pressure,rainfall_rate,thickness_of_rainfall_amount
2019,4,11,0,0,0,1.22,32.75,279.11,76.93,999.40,0.00,0.00
2019,4,11,0,1,0,0.81,321.58,279.06,77.08,999.41,0.00,0.00
2019,4,11,0,2,0,0.52,324.31,279.03,77.17,999.52,0.00,0.00
```

Temperature missing in two out of three measurements:

```csv
year,month,day,hour,minute,second,wind_speed,wind_from_direction,air_temperature,relative_humidity,air_pressure,rainfall_rate,thickness_of_rainfall_amount
2019,4,11,0,0,0,1.22,32.75,,76.93,999.40,0.00,0.00
2019,4,11,0,1,0,0.81,321.58,279.06,77.08,999.41,0.00,0.00
2019,4,11,0,2,0,0.52,324.31,,77.17,999.52,0.00,0.00
```

Wind speed measurements at three heights:

```csv
year,month,day,hour,minute,second,wind_speed(2m),wind_speed(10m),wind_speed(20m)
2019,4,11,0,0,0,1.22,1.21,1.24
2019,4,11,0,1,0,0.81,0.83,0.87
2019,4,11,0,2,0,0.52,0.51,0.50
```

## References

[CF Standard Name Table](https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html)
