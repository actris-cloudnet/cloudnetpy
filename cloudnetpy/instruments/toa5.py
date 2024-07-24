import csv
import datetime
from os import PathLike
from typing import Any


def read_toa5(
    filename: str | PathLike,
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    """Read ASCII data from Campbell Scientific datalogger such as CR1000.

    References:
        CR1000 Measurement and Control System.
        https://s.campbellsci.com/documents/us/manuals/cr1000.pdf
    """
    with open(filename) as file:
        reader = csv.reader(file)
        origin_line = next(reader)
        if len(origin_line) == 0 or origin_line[0] != "TOA5":
            msg = "Invalid TOA5 file"
            raise ValueError(msg)
        header_line = next(reader)
        units_line = next(reader)
        process_line = next(reader)
        output = []
        units = dict(zip(header_line, units_line, strict=False))
        process = dict(zip(header_line, process_line, strict=False))

        row_template: dict[str, Any] = {}
        for header in header_line:
            if "(" in header:
                row_template[header[: header.index("(")]] = []

        for data_line in reader:
            row = row_template.copy()
            for key, value in zip(header_line, data_line, strict=False):
                parsed_value: Any = value
                if key == "TIMESTAMP":
                    parsed_value = datetime.datetime.strptime(
                        parsed_value, "%Y-%m-%d %H:%M:%S"
                    )
                elif key == "RECORD":
                    parsed_value = int(parsed_value)
                if "(" in key:
                    row[key[: key.index("(")]].append(parsed_value)
                else:
                    row[key] = parsed_value
            output.append(row)
        return units, process, output
