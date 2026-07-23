import datetime
from os import PathLike

import numpy as np
import numpy.typing as npt

from cloudnetpy.disdronator.process import DisdroL1
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


def read_rd80_l1(time: npt.NDArray, l0: dict[str, npt.NDArray]) -> DisdroL1:
    return DisdroL1(
        diameter=D,
        diameter_spread=D_SPREAD,
        diameter_bins=D_BINS,
        velocity=V,
        velocity_spread=V_SPREAD,
        velocity_bins=V_BINS,
        time=time,
        interval=l0["Interval [s]"],
        area_nom=AREA_NOM,
        area_eff=None,
        data_raw=l0["n"],
    )


# fmt: off
D = np.array([
    0.3590, 0.4550, 0.5505, 0.6555, 0.7710, 0.9130, 1.1155, 1.3305, 1.5055,
    1.6650, 1.9125, 2.2590, 2.5840, 2.8690, 3.1980, 3.5445, 3.9155, 4.3500,
    4.8590, 5.3725
])
D_BINS = np.array([
    0.313, 0.405, 0.505, 0.596, 0.715, 0.827, 0.999, 1.232, 1.429, 1.582, 1.748,
    2.077, 2.441, 2.727, 3.011, 3.385, 3.704, 4.127, 4.573, 5.145, 5.6
])
D_SPREAD = np.array([
    0.092, 0.100, 0.091, 0.119, 0.112, 0.172, 0.233, 0.197, 0.153, 0.166, 0.329,
    0.364, 0.286, 0.284, 0.374, 0.319, 0.423, 0.446, 0.572, 0.455
])
V = np.array([
    1.435, 1.862, 2.267, 2.692, 3.154, 3.717, 4.382, 4.986, 5.423, 5.793, 6.315,
    7.009, 7.546, 7.903, 8.258, 8.556, 8.784, 8.965, 9.076, 9.137
])  # from manual
V_BINS = np.array([
    1.114, 1.572, 2.042, 2.447, 2.943, 3.379, 3.994, 4.732, 5.280, 5.663, 6.041,
    6.688, 7.269, 7.644, 7.959, 8.299, 8.534, 8.784, 8.987, 9.180, 9.292
])  # using Atlas et al. (1973)
V_SPREAD = np.array([
    0.458, 0.470, 0.405, 0.496, 0.436, 0.615, 0.738, 0.548, 0.383, 0.378, 0.647,
    0.581, 0.375, 0.315, 0.340, 0.235, 0.250, 0.203, 0.193, 0.112
])
AREA_NOM = 5000
# fmt: on
