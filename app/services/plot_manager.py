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

        # cache of the date when the cache was last updated
        self._date_cache: date = date.today()
        # cache of requests
        self._cache: dict[tuple[str, str | None], list[dict]] = {}
        # ie. dict[temporal_axis, specific model or all] = list of dicts of data

        # cache of the date when the kpi cache was last updated
        self._date_kpi_cache: date = date.today()
        # cache of kpi requests
        self._kpi_cache: dict[tuple[str, str | None], list[dict]] = {}
        # ie. dict[temporal_axis, specific model or all] = list of dicts of data

        self._kpi_units_dict: dict[str, str] = {
            "gwp_kgCO2eq": "kgCO2eq",
            "wcf_liters": "l",
            "adpe_mgSbEq": "mgSbEq",
            "energy_kwh": "kwh",
            "total_requests": "requêtes",
        }
        self._kpi_colors_dict: dict[str, str] = {
            "energy_kwh": "#1f77b4",
            "gwp_kgCO2eq": "#d62728",
            "wcf_liters": "#17becf",
            "pd_mj": "#ff7f0e",
            "adpe_mgSbEq": "#9467bd",
        }

        # comparison_dict import logic
        _fp: str = "data/comparison_dict.json"
        self._comparison_dict_path: str = (
            comparison_dict_path if comparison_dict_path else _fp
        )
        try:
            with open(self._comparison_dict_path, "r", encoding="UTF-8") as file:
                self._comparison_dict: dict[str, dict[float, str]] = json.load(file)
                # example :
                # {water: {1000: "un litre d'eau", 72000: "une douche de 6 minutes", ...}}
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Comparison dict file '{self._comparison_dict_path}' not found"
            ) from e

    ################################################################
    # HELPER METHODS : DATA RETRIEVAL
    ################################################################

    def _get_data_grouped_by(
        self, temporal_axis: str, model: str | None = None
    ) -> dict[tuple[str, str | None], list[dict]]:
        # we check if the cache is still valid (ie. same day)
        # otherwise we clear it
        if self._date_cache != date.today():
            self._cache = {}
            self._date_cache = date.today()

        # now we check if the request is already cached
        # if yes we return it
        cache_key = (temporal_axis, tuple(model) if model else "all")
        if cache_key in self._cache:
            return self._cache[cache_key]

        # else we fetch the data
        results = self.llm_usage.get_all_group_by(
            data_grouped_by={"temporal_axis": temporal_axis, "model": model}
        )

        # we cache the result
        self._cache[cache_key] = results
        # we return it
        return results

    def _get_kpi_data_grouped_by(
        self, temporal_axis: str, model: str | None = None
    ) -> dict[tuple[str, str | None], dict]:
        # we check if the kpi cache is still valid (ie. same day)
        # otherwise we clear it
        if self._date_kpi_cache != date.today():
            self._kpi_cache = {}
            self._date_kpi_cache = date.today()

        # now we check if the request is already cached
        cache_key = (temporal_axis, model if model else "all")
        if cache_key in self._kpi_cache:
            return self._kpi_cache[cache_key]

        # else we fetch the data
        results = self.llm_usage.get_aggregated_kpi(
            data_grouped_by={"temporal_axis": temporal_axis, "model": model}
        )

        # we cache the result
        self._kpi_cache[cache_key] = results
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

        kpi_dict: dict[str, str] | None = self._comparison_dict.get(which)
        if kpi_dict is None:
            raise ValueError(f"comparison dict for {which} is None.")

        # mapping stable of float to str keys
        float_to_key = {float(k): k for k in kpi_dict.keys()}
        thresholds = sorted(float_to_key.keys())

        # SPECIAL CASE : total_requests
        if which == "total_requests":
            idx = bisect.bisect_left(thresholds, value)

            # smaller than the first threshold
            if idx == 0:
                ref = thresholds[0]
                pct = round((value / ref) * 100, 2)
                return f"Environ {pct}% du {kpi_dict[float_to_key[ref]]}"

            # else we take the closest threshold
            ref = thresholds[idx] if idx < len(thresholds) else thresholds[-1]
            pct = round((value / ref) * 100, 2)
            return f"Environ {pct}% du {kpi_dict[float_to_key[ref]]}"

        # GENERAL CASE: other KPIs
        idx = bisect.bisect_right(thresholds, value)
        if idx == 0:
            return "Valeur trop faible pour une comparaison."

        lower_key = thresholds[idx - 1]
        ratio = round(value / lower_key, 2)

        # phrase cohérente
        if ratio < 1.1:
            sentence = f"Comparable à {kpi_dict[float_to_key[lower_key]]}"
        else:
            sentence = f"Soit {ratio}x {kpi_dict[float_to_key[lower_key]]}"

        logger.debug("'make_a_comparison' method returning: %s", sentence)
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
        if unit in ["requêtes"]:
            return f"{int(value)} {unit}"
        return f"{round(value, rounded_to)}{unit}"

    def get_kpi_statistic(
        self, which: str, temporal_axis: str, model_name: str | None
    ) -> dict[str, str]:
        """Get KPI statistic for a given KPI, temporal axis and model.

        Args:
            which (str): KPI to retrieve.
            temporal_axis (str): Temporal axis for grouping.
            model_name (str | None): Specific model name or None for all model.

        Raises:
            ValueError: If the specified KPI does not exist.

        Returns:
            dict[str, str]: Dictionary containing formatted KPI value and comparison.
        """
        if which not in self._kpi_units_dict:
            raise ValueError(f"KPI '{which}' does not exist.")

        aggregated = self._get_kpi_data_grouped_by(
            temporal_axis=temporal_axis,
            model=model_name if model_name else None,
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
    # PLOT HELPER METHODS
    ################################################################
    def _get_model_color_map(self, model_names: list[str]) -> dict[str, str]:
        base_palette = [
            "#1f77b4",  # bleu
            "#ff7f0e",  # orange
            "#2ca02c",  # vert
            "#d62728",  # rouge
            "#9467bd",  # violet
            "#8c564b",  # marron
            "#e377c2",  # rose
            "#7f7f7f",  # gris
            "#bcbd22",  # olive
            "#17becf",  # cyan
        ]
        return {
            model: base_palette[i % len(base_palette)]
            for i, model in enumerate(sorted(model_names))
        }

    ################################################################
    # PLOT METHODS
    ################################################################

    def plot_envir_kpis_over_time(
        self,
        temporal_axis: str,
        model_name: str | None = None,
        to_json: bool = True,
    ):
        """Line plots of environmental KPIs over time (energy, CO2, water, etc.), with one line per model."""
        data = self._get_data_grouped_by(
            temporal_axis=temporal_axis,
            model=model_name if model_name else None,
        )[(temporal_axis, model_name)]

        if not data:
            return go.Figure().to_json() if to_json else go.Figure()

        fig = go.Figure()
        model = sorted({d["model_name"] for d in data})

        kpis_to_plot = self._kpi_units_dict.keys() - {"total_requests"}
        for kpi in kpis_to_plot:
            for m in model:
                x = [d["period"] for d in data if d["model_name"] == m]
                y = [d[kpi] for d in data if d["model_name"] == m]
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers",
                        name=f"{kpi} ({m})",
                        line=dict(color=self._kpi_colors_dict[kpi]),
                    )
                )

        fig.update_layout(
            title="KPIs Environnementaux par Modèle",
            xaxis_title="Période",
            yaxis_title="Valeur",
            template="plotly_white",
            legend_title="KPI (Modèle)",
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)

        if to_json:
            return fig.to_json()
        return fig.show()

    def plot_energy_vs_co2(
        self, temporal_axis: str, model_name: str | None = None, to_json: bool = True
    ):
        data_dict = self._get_data_grouped_by(
            temporal_axis=temporal_axis, model=model_name
        )

        fig = go.Figure()

        all_data = []
        for _, data in data_dict.items():
            all_data.extend(data)

        if not all_data:
            return fig.to_json() if to_json else fig

        models = sorted(set(d["model_name"] for d in all_data))
        color_map = self._get_model_color_map(models)

        for model in models:
            model_data = [d for d in all_data if d["model_name"] == model]

            fig.add_trace(
                go.Scatter(
                    x=[d["energy_kwh"] for d in model_data],
                    y=[d["gwp_kgCO2eq"] for d in model_data],
                    mode="markers",
                    name=model,
                    marker=dict(
                        size=12,
                        color=color_map[model],
                        opacity=0.8,
                    ),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "Période : %{customdata}<br>"
                        "Énergie : %{x:.2f} kWh<br>"
                        "CO₂ : %{y:.2f} kgCO₂eq"
                    ),
                    text=[model] * len(model_data),
                    customdata=[d["period"] for d in model_data],
                )
            )

        fig.update_layout(
            title="Diagnostic — Énergie vs CO₂ par modèle",
            xaxis_title="Consommation énergétique (kWh)",
            yaxis_title="Impact carbone (kgCO₂eq)",
            template="plotly_white",
        )

        return fig.to_json() if to_json else fig.show()

    def plot_requests_distribution(
        self, temporal_axis: str, model_name: str | None = None, to_json: bool = True
    ):
        data_dict = self._get_data_grouped_by(
            temporal_axis=temporal_axis, model=model_name
        )

        fig = go.Figure()

        all_data = []
        for _, data in data_dict.items():
            all_data.extend(data)

        if not all_data:
            return fig.to_json() if to_json else fig

        models = sorted(set(d["model_name"] for d in all_data))
        periods = sorted(set(d["period"] for d in all_data))
        color_map = self._get_model_color_map(models)

        for model in models:
            model_data = {d["period"]: d for d in all_data if d["model_name"] == model}

            success_rate = [
                (
                    model_data[p]["total_success"] / model_data[p]["total_requests"]
                    if p in model_data and model_data[p]["total_requests"] > 0
                    else 0
                )
                for p in periods
            ]

            fig.add_trace(
                go.Bar(
                    x=periods,
                    y=success_rate,
                    name=model,
                    marker_color=color_map[model],
                )
            )

        fig.update_layout(
            title="Taux de succès par modèle",
            xaxis_title="Période",
            yaxis_title="Taux de succès",
            barmode="group",
            template="plotly_white",
            yaxis=dict(tickformat=".0%"),
        )

        return fig.to_json() if to_json else fig.show()

    def plot_energy_efficiency(
        self, temporal_axis: str, model_name: str | None = None, to_json: bool = True
    ):
        """
        Bar chart: energy consumption per request (kWh / request),
        grouped by model and period.
        """
        data_dict = self._get_data_grouped_by(
            temporal_axis=temporal_axis, model=model_name
        )

        fig = go.Figure()

        all_data = []
        for _, data in data_dict.items():
            all_data.extend(data)

        if not all_data:
            return fig.to_json() if to_json else fig

        models = sorted(set(d["model_name"] for d in all_data))
        periods = sorted(set(d["period"] for d in all_data))
        color_map = self._get_model_color_map(models)

        for model in models:
            model_data = {d["period"]: d for d in all_data if d["model_name"] == model}

            efficiency = [
                (
                    model_data[p]["energy_kwh"] / model_data[p]["total_requests"]
                    if p in model_data and model_data[p]["total_requests"] > 0
                    else 0
                )
                for p in periods
            ]

            fig.add_trace(
                go.Bar(
                    x=periods,
                    y=efficiency,
                    name=model,
                    marker_color=color_map[model],
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "Période : %{x}<br>"
                        "Énergie / requête : %{y:.4f} kWh"
                    ),
                )
            )

        fig.update_layout(
            title="Efficacité énergétique par requête",
            xaxis_title="Période",
            yaxis_title="kWh / requête",
            barmode="group",
            template="plotly_white",
        )

        return fig.to_json() if to_json else fig.show()

    ################################################################
    # FACADE PATTERN METHODS
    # explanation: Instead of calling all the plot/kpis methods one by one,
    # call this method which will return the result of all of them.
    # link to the pattern: https://en.wikipedia.org/wiki/Facade_pattern
    ################################################################

    def plot_all(
        self,
        temporal_axis: str,
        model_name: str | None = None,
        to_json: bool = True,
    ) -> dict[str, str] | go.Figure:
        data = self._get_data_grouped_by(
            temporal_axis=temporal_axis,
            model=model_name if model_name else None,
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
        args = {
            "temporal_axis": temporal_axis,
            "to_json": to_json,
            "model_name": model_name,
        }
        self.plot_envir_kpis_over_time(**args)
        self.plot_energy_vs_co2(**args)
        self.plot_requests_distribution(**args)
        self.plot_energy_efficiency(**args)

        return plots

    def kpis_all(
        self, temporal_axis: str, model_name: str | None = None
    ) -> dict[str, dict[str, str]]:
        """Get all KPI statistics for a given temporal axis and model.

        Args:
            temporal_axis (str): Temporal axis for grouping.
            model_name (str | None, optional): Specific model name or None for all model. Defaults to None.

        Returns:
            dict[str, dict[str, str]]: Dictionary of KPI statistics.
        """
        data = self._get_kpi_data_grouped_by(
            temporal_axis=temporal_axis,
            model=model_name if model_name else None,
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

    def dummy_data(self, temporal_axis, model_name) -> None:
        """Method to create dummy data for testing purposes."""
        data = self._get_data_grouped_by(
            temporal_axis=temporal_axis,
            model=model_name if model_name else None,
        )
        return data


if __name__ == "__main__":
    # you can test here the class methods
    # BUT FIRST : if you have no data in your db:
    # write in your terminal: python -m scripts.seed_daily_metrics
    # just write in your terminal: python -m app.services.plot_manager
    # arguments for kpis_all : temporal_axis, model_name
    # temporal_axis : "W", "M", "Y"
    # W = week
    # M = month
    # Y = year
    # model_name : "mistral-small" | "mistral-medium" | None
    # None = all models
    temporal_axis = "M"
    model_name = "mistral-medium"
    pm = PlotManager()
    print(pm.dummy_data(temporal_axis, model_name))
    pm.plot_all(temporal_axis, model_name, to_json=False)
    print("kpis : ")
    print(pm.kpis_all(temporal_axis, model_name))
    # have fun :)
