import pytest

from cloudnetpy.model_evaluation.plotting import plotting as pl

MODEL = "ecmwf"


class VariableInfo:
    def __init__(self):
        self.name = "Product"


@pytest.mark.parametrize("key", ["cf_V", "cf_A", "cf_V_adv", "cf_A_adv"])
def test_get_cf_title(key) -> None:
    var = VariableInfo()
    field_name = key + "_" + MODEL
    value = "Product, Volume"
    if "A" in key:
        value = "Product, Area"
    x = pl._get_cf_title(field_name, var)
    assert x == value


@pytest.mark.parametrize("key", ["cf_V", "cf_A", "cf_V_adv", "cf_A_adv"])
def test_get_cf_title_cycle(key) -> None:
    var = VariableInfo()
    field_name = key + "_" + MODEL + "_001"
    value = "Product, Volume"
    if "A" in key:
        value = "Product, Area"
    x = pl._get_cf_title(field_name, var)
    assert x == value


@pytest.mark.parametrize(
    "key, value",
    [
        ("iwc", "Product"),
        ("iwc_att", "Product with good attenuation"),
        ("iwc_rain", "Product with rain"),
        ("iwc_adv", "Product"),
        ("iwc_att_adv", "Product with good attenuation"),
        ("iwc_rain_adv", "Product with rain"),
    ],
)
def test_get_iwc_title(key, value) -> None:
    var = VariableInfo()
    field_name = key + "_" + MODEL
    x = pl._get_iwc_title(field_name, var)
    assert x == value


@pytest.mark.parametrize(
    "key, value",
    [
        ("iwc", "Product"),
        ("iwc_att", "Product with good attenuation"),
        ("iwc_rain", "Product with rain"),
        ("iwc_adv", "Product"),
        ("iwc_att_adv", "Product with good attenuation"),
        ("iwc_rain_adv", "Product with rain"),
    ],
)
def test_get_iwc_title_cycle(key, value) -> None:
    var = VariableInfo()
    field_name = key + "_" + MODEL + "_001"
    x = pl._get_iwc_title(field_name, var)
    assert x == value


def test_get_product_title() -> None:
    var = VariableInfo()
    value = "Product"
    x = pl._get_product_title(var)
    assert x == value


def test_get_product_title_cycle() -> None:
    var = VariableInfo()
    value = "Product"
    x = pl._get_product_title(var)
    assert x == value


@pytest.mark.parametrize(
    "key, title",
    [("lwc", "Product"), ("lwc_adv", "Product (Advection time)")],
)
def test_get_stat_titles(key, title) -> None:
    field_name = key + "_" + MODEL
    var = VariableInfo()
    x = pl._get_stat_titles(field_name, key, var)
    assert x == title


@pytest.mark.parametrize("key", ["cf_V", "cf_A", "cf_V_adv", "cf_A_adv"])
def test_get_cf_title_stat(key) -> None:
    field_name = key + "_" + MODEL
    var = VariableInfo()
    x = pl._get_cf_title_stat(field_name, var)
    value = "Product volume"
    if "A" in key:
        value = "Product area"
    assert x == value


@pytest.mark.parametrize(
    "key, value",
    [
        ("iwc", "Product"),
        ("iwc_att", "Product with good attenuation"),
        ("iwc_rain", "Product with rain"),
    ],
)
def test_get_iwc_title_stat(key, value) -> None:
    field_name = key + "_" + MODEL
    var = VariableInfo()
    x = pl._get_iwc_title_stat(field_name, var)
    assert x == value


@pytest.mark.parametrize("key", ["lwc"])
def test_get_product_title_stat(key) -> None:
    var = VariableInfo()
    x = pl._get_product_title_stat(var)
    assert x == "Product"
