import bisect
import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta

import numpy as np
from plotly import graph_objects as go

from app.schemas.llm_metrics_schema import LLMMetricsSchema
from app.services.llm_usage_service import LLMUsageService

logger = logging.getLogger(__name__)


class PlotManager:
    def __init__(
        self,
        llm_usage: LLMUsageService = LLMUsageService(),
        comparison_dict_path: str | None = None,
    ) -> None:
        """Initialize the PlotManager..

        Args:
            usage_service (LLMUsageService): LLM usage service instance.
        """
        self.llm_usage = llm_usage

        # cache of requests
        self._cache: dict[tuple[str, str | None], list[dict]] = {}
        # ie. dict[temporal_axis, specific model or all] = list of dicts of data
        self._date_cache: date = date.today()

        self._kpi_units_dict: dict[str, str] = {
            "gwp_kgCO2eq": "kgCO2eq",
            "wcf_liters": "l",
            "adpe_mgSbEq": "mgSbEq",
            "energy_kwh": "kwh",
            "total_requests": "nombre de requêtes",
        }

        # # comparison_dict import logic
        # _fp: str = "data/comparison_dict.json"
        # self._comparion_dict_path: str = (
        #     comparison_dict_path if comparison_dict_path else _fp
        # )
        # try:
        #     with open(self._comparion_dict_path, "r", encoding="UTF-8") as file:
        #         self._comparison_dict: dict[str, dict[float, str]] = json.load(file)
        #         # example :
        #         # {water: {1000: "un litre d'eau", 72000: "une douche de 6 minutes", ...}}
        # except FileNotFoundError as e:
        #     raise FileNotFoundError(
        #         f"Comparison dict file '{self._comparion_dict_path}' not found"
        #     ) from e
        self.comparison_dict = {
            "water": {
                1000: "un litre d'eau",
                72000: "une douche de 6 minutes",
                150000: "un bain",
                2000000: "la consommation moyenne journalière d'une personne en France",
            },
            "gwp_kgCO2eq": {
                0.21: "un kilomètre en voiture thermique",
                0.05: "un kilomètre en vélo",
                0.012: "un kilomètre en train",
                0.4: "une heure de visioconférence",
            },
            "energy_kwh": {
                0.1: "une heure d'ordinateur portable",
                0.5: "une heure de télévision",
                1.5: "une machine à laver",
                3.0: "un sèche-linge",
            },
        }

    ################################################################
    # HELPER METHODS : DATA RETRIEVAL
    ################################################################

    def _get_data_grouped_by(
        self, temporal_axis: str, models: list[str] | None = None
    ) -> dict[tuple[str, str | None], list[dict]]:
        # we check if the cache is still valid (ie. same day)
        # otherwise we clear it
        if self._date_cache != date.today():
            self._cache = {}
            self._date_cache = date.today()

        # now we check if the request is already cached
        # if yes we return it
        cache_key = (temporal_axis, tuple(models) if models else "all")
        if cache_key in self._cache:
            return self._cache[cache_key]

        # else we fetch the data
        results = self.llm_usage.get_all_group_by(
            {"temporal_axis": temporal_axis, "model": models}
        )
        # we cache the result
        self._cache[cache_key] = results
        # we return it
        return results

    ################################################################
    # KPI METHODS
    ################################################################

    def make_a_comparison(self, which: str, value: float) -> str:
        """Make a comparison with a "real-world" action / object, for a given KPI and value.

        Args:
            which (str): KPI to compare.
            value (float): Value of the KPI.
        Raises:
            ValueError: If the comparison dictionary for the specified KPI is not found.

        Returns:
            str: Comparison sentence.
        """
        logger.debug("'make_a_comparison' method called.")
        kpi_dict: dict[float, str] | None = self._comparison_dict.get(which, None)
        if kpi_dict is None:
            raise ValueError(f"comparison dict for {which} is None.")

        # we get the number just lower or equal to the value provided
        # Carefull thought: bisect assumes the orderable-like first arg is sorted ASC.
        thresholds: list[float] = sorted(kpi_dict.keys())
        idx = bisect.bisect_right(thresholds, value)
        if idx == 0:
            return "Valeur trop faible pour une comparaison."

        lower_key: float = thresholds[idx - 1]
        # now that we have the lower number and the value, we calculate the ratio
        ratio: float = round(value / lower_key, 2)

        # finally we make a sentence and return it
        sentence: str = f"Soit {ratio}x {kpi_dict[lower_key]}"
        logger.debug("'make_a_comparison' method returning : %s", sentence)
        return sentence

    def _format_kpi_value(self, value: float, unit: str, rounded_to: int = 2) -> str:
        """Format KPI value with unit.

        Args:
            value (float): Value to format.
            unit (str): Unit of the value.
            rounded_to (int, optional): Number of decimal places to round to. Defaults to 2.

        Returns:
            str: Formatted KPI value with unit.
        """
        return f"{round(value, rounded_to)}{unit}"

    def get_kpi_statistic(
        self, which: str, temporal_axis: str, model_name: str | None
    ) -> dict[str, str]:
        """Get KPI statistic for a given KPI, temporal axis and model.

        Args:
            which (str): KPI to retrieve.
            temporal_axis (str): Temporal axis for grouping.
            model_name (str | None): Specific model name or None for all models.

        Raises:
            ValueError: If the specified KPI does not exist.

        Returns:
            dict[str, str]: Dictionary containing formatted KPI value and comparison.
        """
        if which not in self._kpi_units_dict:
            raise ValueError(f"KPI '{which}' does not exist.")

        aggregated = self._get_data_grouped_by(
            temporal_axis=temporal_axis,
            models=[model_name] if model_name else None,
        )[(temporal_axis, model_name)]

        value = aggregated[which]
        unit = self._kpi_units_dict[which]

        formatted_value = self._format_kpi_value(value, unit)
        comparison = self.make_a_comparison(which, value)

        return {
            "value": formatted_value,
            "comparison": comparison,
        }

    ################################################################
    # PLOT METHODS
    ################################################################

    def _get_model_palette(self, models: list[str]) -> dict[str, str]:
        """
        Return a color mapping for each model.
        Colors are chosen from a Plotly qualitative palette and repeated if necessary.
        """
        # Qualitative Plotly colors (10 max, will repeat if more models)
        base_colors = [
            "#636EFA",
            "#EF553B",
            "#00CC96",
            "#AB63FA",
            "#FFA15A",
            "#19D3F3",
            "#FF6692",
            "#B6E880",
            "#FF97FF",
            "#FECB52",
        ]
        palette = {}
        for i, model in enumerate(sorted(models)):
            palette[model] = base_colors[i % len(base_colors)]
        return palette

    ################################################################
    # FACADE PATTERN METHODS
    # explanation: Instead of calling all the plot/kpis methods one by one,
    # call this method which will return the result of all of them.
    # link to the pattern: https://en.wikipedia.org/wiki/Facade_pattern
    ################################################################

    def plot_all(self, temporal_axis: str, model_name: str | None = None):
        data = self._get_data_grouped_by(
            temporal_axis=temporal_axis,
            models=[model_name] if model_name else None,
        )[(temporal_axis, model_name)]
        if data == []:
            logger.info(
                "No data found for plot_all with temporal_axis=%s and model_name=%s",
                temporal_axis,
                model_name,
            )
            return {}

        # now that we have data, we dispatch the data to the different plot methods
        plots: dict[str, str] | None = None
        # plots:  {name_of_plot: __json_string__}
        # -> dict of plots

        return plots

    def kpis_all(
        self, temporal_axis: str, model_name: str | None = None
    ) -> dict[str, dict[str, str]]:
        """Get all KPI statistics for a given temporal axis and model.

        Args:
            temporal_axis (str): Temporal axis for grouping.
            model_name (str | None, optional): Specific model name or None for all models. Defaults to None.

        Returns:
            dict[str, dict[str, str]]: Dictionary of KPI statistics.
        """
        data = self._get_data_grouped_by(
            temporal_axis=temporal_axis,
            models=[model_name] if model_name else None,
        )[(temporal_axis, model_name)]
        if data == []:
            logger.info(
                "No data found for kpis_all with temporal_axis=%s and model_name=%s",
                temporal_axis,
                model_name,
            )
            return {}

        return {
            kpi: self.get_kpi_statistic(kpi, temporal_axis, model_name)
            for kpi in self._kpi_units_dict
        }


if __name__ == "__main__":
    pm = PlotManager()
    print(pm.kpis_all("W", None))
