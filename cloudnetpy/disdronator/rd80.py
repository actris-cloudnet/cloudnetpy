import datetime
from os import PathLike

import numpy as np
import numpy.typing as npt

from cloudnetpy.disdronator.utils import convert_to_numpy


def read_rd80(filename: str | PathLike) -> tuple[npt.NDArray, dict[str, npt.NDArray]]:
    time = []
    data: dict[str, list] = {
        "n": [],
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
            data["Interval [s]"].append(int(row["Interval [s]"]))
            for key in ("RI [mm/h]", "RA [mm]", "RAT [mm]"):
                data[key].append(float(row[key].replace(",", ".")))
    return np.array(time), convert_to_numpy(data)


# fmt: off
Dmid = np.array([
    0.359, 0.455, 0.551, 0.656, 0.771, 0.913, 1.116, 1.331, 1.506, 1.665, 1.912,
    2.259, 2.584, 2.869, 3.198, 3.544, 3.916, 4.350, 4.859, 5.373
])  # mm
Dspr = np.array([
    0.092, 0.100, 0.091, 0.119, 0.112, 0.172, 0.233, 0.197, 0.153, 0.166, 0.329,
    0.364, 0.286, 0.284, 0.374, 0.319, 0.423, 0.446, 0.572, 0.455
])  # mm
Dlow = np.array([
    0.313, 0.405, 0.505, 0.596, 0.715, 0.827, 0.999, 1.232, 1.429, 1.582, 1.748,
    2.077, 2.441, 2.727, 3.011, 3.385, 3.704, 4.127, 4.573, 5.145
])  # mm
Dupp = np.append(Dlow[1:], Dlow[-1] + Dspr[-1])  # mm
Vmid = np.array([
    1.435, 1.862, 2.267, 2.692, 3.154, 3.717, 4.382, 4.986, 5.423, 5.793, 6.315,
    7.009, 7.546, 7.903, 8.258, 8.556, 8.784, 8.965, 9.076, 9.137
])  # m s-1
A = 0.005  # m2
# fmt: on
