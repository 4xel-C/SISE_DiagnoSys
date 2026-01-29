import logging
from collections import defaultdict
from datetime import datetime, date, timedelta
import bisect
import json
import numpy as np
from plotly import graph_objects as go

from app.schemas.llm_metrics_schema import LLMMetricsSchema
from app.services.llm_usage_service import LLMUsageService

logger = logging.getLogger(__name__)

class PlotManager:
    def __init__(self, llm_usage: LLMUsageService = LLMUsageService(), comparison_dict_path: str | None = None) -> None:
        """Initialize the PlotManager..

        Args:
            usage_service (LLMUsageService): LLM usage service instance.
        """
        self.llm_usage = llm_usage

        # cache of requests
        self._cache: dict[
            tuple[str, list[str] | str],
            list[LLMMetricsSchema]
        ] = {}
        # ie. dict[number of days, list of models or all] = list of schemas

        # cache of aggregated metrics
        self._agg_cache: dict[
            tuple[str, str, str],
            dict[tuple[date, str], float]
        ] = {}
        # ie. dict[temporal_axis, model, metric] = dict[(period_date, model_name), value]

        # cache of aggregated requets
        self._kpi_cache: dict[
            tuple[str, str | None],
            dict[str, float]
        ] = {}
        # ie. dict[number of days, list of models or all] = dict[kpi, value]

        self._kpi_units_dict: dict[str, str] = {
            "gwp_kgCO2eq": "kgCO2eq",
            "wcf_liters": "l",
            "adpe_mgSbEq": "mgSbEq",
            "energy_kwh": "kwh",
            "total_requests": "nombre de requêtes",
        }

        # comparison_dict import logic
        _fp: str = "data/comparison_dict.json"
        self._comparion_dict_path: str = comparison_dict_path if comparison_dict_path else _fp
        try:
            with open(self._comparion_dict_path, "r", encoding="UTF-8") as file:
                self._comparison_dict: dict[str, dict[float, str]] = json.load(file)
                # example :
                # {water: {1000: "un litre d'eau", 72000: "une douche de 6 minutes", ...}}
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Comparison dict file '{self._comparion_dict_path}' not found"
                ) from e

    ################################################################
    # HELPER METHODS : DATA RETRIEVAL
    ################################################################

    def _get_aggregated_metric(
        self,
        temporal_axis: str,
        metric: str,
        model: str | None = None,
    ) -> dict[tuple[date, str], float]:
        """
        Return aggregated metric by period and model as {(period_date, model_name): value}.
        Stores the result in a dedicated aggregation cache.
        """
        if model is None:
            model = "all"

        # Check aggregation cache
        agg_cache_key = (temporal_axis, model, metric)
        if agg_cache_key in self._agg_cache:
            return self._agg_cache[agg_cache_key]

        # Retrieve raw data (from cache or service)
        cache_key = (temporal_axis, model)
        if cache_key in self._cache and self._cache[cache_key]:
            data: list[LLMMetricsSchema] = self._cache[cache_key]
        else:
            data = self.llm_usage.get_all()
            self._cache[cache_key] = data

        # Perform aggregation
        result: dict[tuple[date, str], float] = defaultdict(float)
        for row in data:
            if model not in ("all", row.nom_modele):
                continue

            d = row.usage_date
            if temporal_axis == "W":
                year, week, _ = d.isocalendar()
                period_date = date.fromisocalendar(year, week, 1)
            elif temporal_axis == "M":
                period_date = date(d.year, d.month, 1)
            elif temporal_axis == "Y":
                period_date = date(d.year, 1, 1)
            else:
                raise ValueError("temporal_axis must be one of W, M, Y")

            result[(period_date, row.nom_modele)] += getattr(row, metric) or 0.0

        # Store aggregated result in cache
        self._agg_cache[agg_cache_key] = result

        return result

    ################################################################
    # KPI METHODS
    ################################################################

    def _aggregate_kpis(
        self,
        temporal_axis: str,
        model_name: str | None,
    ) -> dict[str, float]:
        cache_key = (temporal_axis, model_name)
        if cache_key in self._kpi_cache:
            logger.debug("KPI cache hit for %s", cache_key)
            return self._kpi_cache[cache_key]

        logger.debug("KPI cache miss for %s", cache_key)

        totals: dict[str, float] = {}

        for kpi in self._kpi_units_dict.keys():
            # Use the aggregation cache per KPI
            agg = self._get_aggregated_metric(temporal_axis, kpi, model_name)
            total = sum(agg.values())
            totals[kpi] = total

        # Store in KPI cache
        self._kpi_cache[cache_key] = totals
        return totals

    def make_a_comparison(self, which:str, value: float) -> str:
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
        return f"{round(value, rounded_to)}{unit}"

    def get_kpi_statistic(
        self,
        which: str,
        temporal_axis: str,
        model_name: str | None,
        data: list[LLMMetricsSchema],
    ) -> dict[str, str]:

        if which not in self._kpi_units_dict:
            raise ValueError(f"KPI '{which}' does not exist.")

        aggregated = self._aggregate_kpis(
            temporal_axis=temporal_axis,
            model_name=model_name,
            data=data,
        )

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
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
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
        data: list[LLMMetricsSchema] = self._get_data_for(temporal_axis=temporal_axis, model=model_name)

        # now that we have data, we dispatch the data to the different plot methods
        plots: dict[str, str] | None = None
        # plots:  {name_of_plot: __json_string__}
        # -> dict of plots

        return plots

    def kpis_all(self, temporal_axis: str, model_name: str | None = None):
        data = self._get_data_for(temporal_axis=temporal_axis, model=model_name)

        return {
            kpi: self.get_kpi_statistic(
                which=kpi,
                temporal_axis=temporal_axis,
                model_name=model_name,
                data=data,
            )
            for kpi in self._kpi_units_dict
        }

