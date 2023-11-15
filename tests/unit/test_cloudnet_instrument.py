from tempfile import NamedTemporaryFile

import netCDF4
import pytest

from cloudnetpy.instruments.cloudnet_instrument import CloudnetInstrument


@pytest.mark.parametrize(
    "data, result",
    [
        ("23", 23.0),
        ("23.0", 23.0),
        ("23.0m", 23.0),
        ("23.0 m", 23.0),
    ],
)
def test_parse_global_attribute_numeral(data: str, result: float):
    instrument = CloudnetInstrument()
    with netCDF4.Dataset(
        NamedTemporaryFile().name,
        "w",
        format="NETCDF4_CLASSIC",
    ) as nc:
        instrument.dataset = nc
        for key in ["Altitude", "Latitude", "Longitude"]:
            setattr(nc, key, data)
            assert instrument.parse_global_attribute_numeral(key) == result
