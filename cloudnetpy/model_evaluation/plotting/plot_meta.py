from typing import NamedTuple


class PlotMeta(NamedTuple):
    name: str
    cbar: str
    plot_range: tuple
    plot_scale: str
    plot_type: str
    clabel: str | None = None
    hist_bin: int | None = None
    x_title: str | None = None
    ylabel: str | None = None
    hist_limits: tuple | None = None


_LOG = "logarithmic"
_LIN = "linear"

_M3 = "$m^{-3}$"
_MS1 = "m s$^{-1}$"
_SR1M1 = "sr$^{-1}$ m$^{-1}$"
_KGM2 = "kg m$^{-2}$"
_KGM3 = "kg m$^{-3}$"
_KGM2S1 = "kg m$^{-2}$ s$^{-1}$"


ATTRIBUTES = {
    "drizzle_N": PlotMeta(
        name="Drizzle number concentration",
        cbar="viridis",
        clabel=_M3,
        plot_range=(1e4, 1e9),
        plot_scale=_LIN,
        plot_type="mesh",
    ),
    "v_air": PlotMeta(
        name="Vertical air velocity",
        cbar="RdBu_r",
        clabel=_MS1,
        plot_range=(-2, 2),
        plot_scale=_LIN,
        plot_type="mesh",
    ),
    "uwind": PlotMeta(
        name="Model zonal wind",
        cbar="RdBu_r",
        clabel=_MS1,
        plot_range=(-50, 50),
        plot_scale=_LIN,
        plot_type="model",
    ),
    "vwind": PlotMeta(
        name="Model meridional wind",
        cbar="RdBu_r",
        clabel=_MS1,
        plot_range=(-50, 50),
        plot_scale=_LIN,
        plot_type="model",
    ),
    "temperature": PlotMeta(
        name="Model temperature",
        cbar="RdBu_r",
        clabel="K",
        plot_range=(223.15, 323.15),
        plot_scale=_LIN,
        plot_type="model",
    ),
    "specific_humidity": PlotMeta(
        name="Model specific humidity",
        cbar="viridis",
        clabel="",
        plot_range=(1e-5, 1e-2),
        plot_scale=_LIN,
        plot_type="model",
    ),
    "q": PlotMeta(
        name="Model specific humidity",
        cbar="viridis",
        clabel="",
        plot_range=(1e-5, 1e-2),
        plot_scale=_LIN,
        plot_type="model",
    ),
    "pressure": PlotMeta(
        name="Model pressure",
        cbar="viridis",
        clabel="Pa",
        plot_range=(1e4, 1.5e5),
        plot_scale=_LIN,
        plot_type="model",
    ),
    "v": PlotMeta(
        name="Doppler velocity",
        cbar="RdBu_r",
        clabel=_MS1,
        plot_range=(-4, 4),
        plot_scale=_LIN,
        plot_type="mesh",
    ),
    "lwp": PlotMeta(
        name="Liquid water path",
        cbar="Blues",
        ylabel=_KGM2,
        plot_range=(0, 1),
        plot_scale=_LIN,
        plot_type="bar",
    ),
    "cf": PlotMeta(
        name="Cloud fraction",
        cbar="Blues",
        clabel="",
        plot_range=(0, 1),
        plot_scale=_LIN,
        plot_type="model",
        hist_bin=10,
        hist_limits=(0.0, 1.1, 0.1),
        x_title="",
    ),
    "cf_cirrus": PlotMeta(
        name="Cloud fraction with filtered Cirrus",
        cbar="Blues",
        clabel="",
        plot_range=(0, 1),
        plot_scale=_LIN,
        plot_type="model",
        hist_bin=10,
        hist_limits=(0.0, 1.1, 0.1),
        x_title="",
    ),
    "iwc": PlotMeta(
        name="Ice water content",
        cbar="viridis",
        clabel=_KGM3,
        plot_range=(1e-7, 1e-3),
        plot_scale=_LOG,
        plot_type="mesh",
        hist_bin=11,
        hist_limits=(0.0, 3.4e-5, 0.3e-5),
        x_title="g/kg",
    ),
    "lwc": PlotMeta(
        name="Liquid water content",
        cbar="Blues",
        clabel=_KGM3,
        plot_range=(1e-5, 1e-2),
        plot_scale=_LOG,
        plot_type="mesh",
        hist_bin=10,
        hist_limits=(0.0, 3.4e-5, 0.3e-5),
        x_title="g/kg",
    ),
}
