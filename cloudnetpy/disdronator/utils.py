from collections.abc import Container

import numpy as np
from numpy import ma


def convert_to_numpy(
    data: dict,
    fill_values: dict | None = None,
    int_keys: Container | None = None,
    float_keys: Container | None = None,
) -> dict:
    output = {}
    for key, value in data.items():
        arr = np.array(value)
        if fill_values is not None and key in fill_values:
            arr = ma.masked_where(arr == fill_values[key], arr)
        if int_keys is not None and key in int_keys:
            new_arr = arr.astype(np.int32)
            if not np.all(arr == new_arr):
                msg = "Cannot convert non-integer float to integer"
                raise ValueError(msg)
            arr = new_arr
        elif float_keys is not None and key in float_keys:
            arr = arr.astype(np.float32)
        output[key] = arr
    return output
