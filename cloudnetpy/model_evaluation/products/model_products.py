import importlib
import logging

import numpy as np
from numpy import ma

from cloudnetpy.datasource import DataSource
from cloudnetpy.model_evaluation.model_metadata import MODELS, VARIABLES
from cloudnetpy.model_evaluation.utils import file_exists
from cloudnetpy.utils import isscalar


class ModelManager(DataSource):
    """Class to collect and manage model data.

    Args:
        model_file (str): Path to source model file.
        model (str): Name of model
        output_file (str): name of output file name and path to save data
        product (str): name of product to generate

    Notes:
        For this class to work, needed information of model in use should be found in
        model_metadata.py

        Output_file is given for saving all cycles to same nc-file. Some variables
        are same in control run and cycles so checking existence of output-file
        prevents duplicates as well as unnecessary processing.

        Class inherits DataSource interface from CloudnetPy.
    """

    def __init__(
        self,
        model_file: str,
        model: str,
        output_file: str,
        product: str,
        *,
        check_file: bool = True,
    ):
        super().__init__(model_file)
        self.model = model
        self.model_info = MODELS[model]
        self.model_vars = VARIABLES["variables"]
        self._product = product
        self.keys: dict = {}
        self._is_file = file_exists(output_file) if check_file else False
        self.cycle = self._read_cycle_name(model_file)
        self._add_variables()
        self._generate_products()
        self.date: list = []
        self.wind = self._calculate_wind_speed()
        self.resolution_h = self._get_horizontal_resolution()

    def _read_cycle_name(self, model_file: str) -> str:
        """Get cycle name from model_metadata.py for saving variable name(s)."""
        try:
            cycles = self.model_info.cycle
            if cycles is None:
                return ""
            cycles_split = [x.strip() for x in cycles.split(",")]
            for cycle in cycles_split:
                if cycle in model_file:
                    return f"_{cycle}"
        except AttributeError:
            return ""
        return ""

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
        cf_name = self.get_model_var_names(("cf",))[0]
        cf = self.getvar(cf_name)
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
        var = []
        for arg in args:
            var.append(VARIABLES[arg].long_name)
        return var

    def get_water_content(self, var: str) -> np.ndarray:
        p_name = self.get_model_var_names(("p",))[0]
        t_name = self.get_model_var_names(("T",))[0]
        lwc_name = self.get_model_var_names((var,))[0]
        p = self.getvar(p_name)
        t = self.getvar(t_name)
        q = self.getvar(lwc_name)
        wc = self._calc_water_content(q, p, t)
        wc = self.cut_off_extra_levels(wc)
        wc[wc < 0.0] = ma.masked
        return wc

    @staticmethod
    def _calc_water_content(q: np.ndarray, p: np.ndarray, t: np.ndarray) -> np.ndarray:
        return q * p / (287 * t)

    def _add_variables(self) -> None:
        """Add basic variables off model and cycle."""

        def _add_common_variables() -> None:
            """Model variables that are always the same within cycles."""
            wanted_vars = self.model_vars.common_var
            if wanted_vars is None:
                msg = f"Model {self.model} has no common variables"
                raise ValueError(msg)
            wanted_vars_split = [x.strip() for x in wanted_vars.split(",")]
            for var in wanted_vars_split:
                if var in self.dataset.variables:
                    data = self.dataset.variables[var][:]
                    if not isscalar(data) and len(data) > 25:
                        data = self.cut_off_extra_levels(self.dataset.variables[var][:])
                    self.append_data(data, f"{var}")

        def _add_cycle_variables() -> None:
            """Add cycle depending variables."""
            wanted_vars = self.model_vars.cycle_var
            if wanted_vars is None:
                msg = f"Model {self.model} has no cycle variables"
                raise ValueError(msg)
            wanted_vars_split = [x.strip() for x in wanted_vars.split(",")]
            for var in wanted_vars_split:
                if var in self.dataset.variables:
                    data = self.dataset.variables[var][:]
                    if data.ndim > 1 or len(data) > 25:
                        data = self.cut_off_extra_levels(self.dataset.variables[var][:])
                    self.append_data(data, f"{self.model}{self.cycle}_{var}")
                if var == "height":
                    self.keys["height"] = f"{self.model}{self.cycle}_{var}"

        if not self._is_file:
            _add_common_variables()
        _add_cycle_variables()

    def cut_off_extra_levels(self, data: np.ndarray) -> np.ndarray:
        """Remove unused levels (over 22km) from model data."""
        try:
            level = self.model_info.level
        except KeyError:
            return data

        return data[:, :level] if data.ndim > 1 else data[:level]

    def _calculate_wind_speed(self) -> np.ndarray:
        """Real wind from x- and y-components."""
        u = self.getvar("uwind")
        v = self.getvar("vwind")
        u = self.cut_off_extra_levels(u)
        v = self.cut_off_extra_levels(v)
        return np.sqrt(ma.power(u.data, 2) + ma.power(v.data, 2))

    def _get_horizontal_resolution(self) -> float:
        h_res = self.getvar("horizontal_resolution")
        return float(np.unique(h_res.data)[0])
