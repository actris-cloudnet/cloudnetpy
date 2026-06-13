import importlib
import logging
from os import PathLike

import numpy as np
import numpy.typing as npt
from numpy import ma

from cloudnetpy.datasource import DataSource
from cloudnetpy.exceptions import ModelDataError
from cloudnetpy.model_evaluation.model_metadata import (
    ALTITUDE_LIMIT,
    COMMON_VARIABLES,
    CYCLE_VARIABLES,
    MODEL_VARIABLE_NAMES,
)
from cloudnetpy.model_evaluation.utils import file_exists


class ModelManager(DataSource):
    """Class to collect and manage model data.

    Args:
        model_file (str): Path to source model file.
        model (str): Name of model
        output_file (str): name of output file name and path to save data
        product (str): name of product to generate

    Notes:
        Model files are harmonized by the model munger, so all models share the
        same variable names and units. The only model-specific quantity is the
        number of vertical levels, which is derived from the model height and
        the shared `ALTITUDE_LIMIT`.

        Output_file is given for saving all cycles to same nc-file. Some variables
        are same in control run and cycles so checking existence of output-file
        prevents duplicates as well as unnecessary processing.

        Class inherits DataSource interface from CloudnetPy.
    """

    def __init__(
        self,
        model_file: str | PathLike,
        model: str,
        output_file: str | PathLike,
        product: str,
        *,
        check_file: bool = True,
    ) -> None:
        super().__init__(model_file)
        self.model = model
        self._product = product
        self.keys: dict = {}
        self._is_file = file_exists(output_file) if check_file else False
        self.cycle = ""
        self._n_levels = self._read_number_of_levels()
        self._add_variables()
        self._generate_products()
        self.date: list = []
        self.wind = self._calculate_wind_speed()
        self.resolution_h = self._get_horizontal_resolution()

    def _read_number_of_levels(self) -> int | None:
        """Number of vertical levels below the altitude limit.

        Model heights are ground-first and increase with level index, and the
        number of levels below the limit is constant in time, so a single level
        count can be used to drop the unused upper levels for any model.
        """
        if "height" not in self.dataset.variables:
            return None
        height = self.to_m(self.dataset.variables["height"])
        below_limit = (
            np.any(height < ALTITUDE_LIMIT, axis=0)
            if height.ndim > 1
            else height < ALTITUDE_LIMIT
        )
        return int(np.sum(below_limit))

    def _generate_products(self) -> None:
        """Process needed data of model to a ModelManager object."""
        cls = importlib.import_module(__name__).ModelManager
        try:
            name = f"_get_{self._product}"
            getattr(cls, name)(self)
        except AttributeError:
            msg = f"Invalid product name: {self._product}"
            logging.exception(msg)
            raise

    def _get_cf(self) -> None:
        """Collect cloud fraction straight from model file."""
        cf = self._getvar_checked("cf")
        cf = self.cut_off_extra_levels(cf)
        cf[cf < 0.05] = ma.masked
        self.append_data(cf, f"{self.model}{self.cycle}_cf")
        self.keys[self._product] = f"{self.model}{self.cycle}_cf"

    def _get_iwc(self) -> None:
        iwc = self.get_water_content("iwc")
        iwc[iwc < 1e-7] = ma.masked
        self.append_data(iwc, f"{self.model}{self.cycle}_iwc")
        self.keys[self._product] = f"{self.model}{self.cycle}_iwc"

    def _get_lwc(self) -> None:
        lwc = self.get_water_content("lwc")
        lwc[lwc < 1e-5] = ma.masked
        self.append_data(lwc, f"{self.model}{self.cycle}_lwc")
        self.keys[self._product] = f"{self.model}{self.cycle}_lwc"

    @staticmethod
    def get_model_var_names(args: tuple) -> list:
        return [MODEL_VARIABLE_NAMES[arg] for arg in args]

    def _getvar_checked(self, *internal_keys: str) -> npt.NDArray:
        """Fetch a model variable, raising a clear error if it is missing."""
        names = [MODEL_VARIABLE_NAMES[key] for key in internal_keys]
        try:
            return self.getvar(*names)
        except KeyError as err:
            msg = (
                f"Model '{self.model}' is missing variable '{names[0]}' "
                f"required for product '{self._product}'."
            )
            raise ModelDataError(msg) from err

    def get_water_content(self, var: str) -> npt.NDArray:
        p = self._getvar_checked("p")
        t = self._getvar_checked("T")
        q = self._getvar_checked(var)
        wc = self._calc_water_content(q, p, t)
        wc = self.cut_off_extra_levels(wc)
        wc[wc < 0.0] = ma.masked
        return wc

    @staticmethod
    def _calc_water_content(
        q: npt.NDArray, p: npt.NDArray, t: npt.NDArray
    ) -> npt.NDArray:
        return q * p / (287 * t)

    def _add_variables(self) -> None:
        """Add basic variables off model and cycle."""

        def _add_variable(var: str, key: str) -> None:
            ncvar = self.dataset.variables[var]
            data = ncvar[:]
            if "level" in ncvar.dimensions:
                data = self.cut_off_extra_levels(data)
            self.append_data(data, key)

        def _add_common_variables() -> None:
            """Model variables that are always the same within cycles."""
            for var in COMMON_VARIABLES:
                if var in self.dataset.variables:
                    _add_variable(var, var)

        def _add_cycle_variables() -> None:
            """Add cycle depending variables."""
            for var in CYCLE_VARIABLES:
                if var in self.dataset.variables:
                    _add_variable(var, f"{self.model}{self.cycle}_{var}")
                if var == "height":
                    self.keys["height"] = f"{self.model}{self.cycle}_{var}"

        if not self._is_file:
            _add_common_variables()
        _add_cycle_variables()

    def cut_off_extra_levels(self, data: npt.NDArray) -> npt.NDArray:
        """Remove unused levels (above the altitude limit) from model data."""
        if self._n_levels is None:
            return data
        return data[:, : self._n_levels] if data.ndim > 1 else data[: self._n_levels]

    def _calculate_wind_speed(self) -> npt.NDArray:
        """Real wind from x- and y-components."""
        u = self.getvar("uwind")
        v = self.getvar("vwind")
        u = self.cut_off_extra_levels(u)
        v = self.cut_off_extra_levels(v)
        return np.sqrt(ma.power(u.data, 2) + ma.power(v.data, 2))

    def _get_horizontal_resolution(self) -> float:
        try:
            h_res = self.getvar("horizontal_resolution")
        except KeyError as err:
            msg = (
                f"Model '{self.model}' is missing 'horizontal_resolution'. "
                "It needs to be added to the model file by the model munger."
            )
            raise ModelDataError(msg) from err
        return float(np.unique(h_res.data)[0])
