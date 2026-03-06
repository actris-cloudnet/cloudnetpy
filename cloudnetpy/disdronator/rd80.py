import datetime
from os import PathLike


def read_rd80(filename: str | PathLike) -> tuple[list, dict[str, list]]:
    time = []
    data: dict[str, list] = {
        "n": [],
        "Status": [],
        "Interval [s]": [],
        "RI [mm/h]": [],
        "RA [mm]": [],
        "RAT [mm]": [],
    }
    with open(filename) as f:
        keys = f.readline().rstrip("\r\n").split("\t")
        for line in f:
            try:
                row = dict(zip(keys, line.rstrip("\r\n").split("\t"), strict=True))
            except ValueError:
                continue
            dt = datetime.datetime.strptime(
                f"{row['YYYY-MM-DD']} {row['hh:mm:ss']}", "%Y-%m-%d %H:%M:%S"
            )
            time.append(dt)
            data["n"].append([int(row[f"n{i + 1}"]) for i in range(20)])
            data["Status"].append(row["Status"])
            data["Interval [s]"].append(int(row["Interval [s]"]))
            for key in ("RI [mm/h]", "RA [mm]", "RAT [mm]"):
                data[key].append(float(row[key].replace(",", ".")))
    return time, data
