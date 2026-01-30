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

    def plot_energy_vs_co2(self, temporal_axis: str, to_json: bool = True):
        data_dict = self._get_data_grouped_by(temporal_axis=temporal_axis, model=None)

        fig = go.Figure()

        for _, data in data_dict.items():
            for d in data:
                fig.add_trace(
                    go.Scatter(
                        x=[d["energy_kwh"]],
                        y=[d["gwp_kgCO2eq"]],
                        mode="markers",
                        marker=dict(
                            size=12, color=self._kpi_colors_dict["gwp_kgCO2eq"]
                        ),
                        name=d["model_name"],
                        hovertemplate=(
                            "Modèle: %{text}<br>"
                            "Énergie: %{x:.2f} kWh<br>"
                            "CO₂: %{y:.2f} kg"
                        ),
                        text=[d["model_name"]],
                        showlegend=False,
                    )
                )

        fig.update_layout(
            title="Diagnostic — Énergie vs CO₂",
            xaxis_title="Énergie (kWh)",
            yaxis_title="CO₂ (kgCO₂eq)",
            template="plotly_white",
        )

        return fig.to_json() if to_json else fig.show()

    def plot_requests_distribution(self, temporal_axis: str, to_json: bool = True):
        data_dict = self._get_data_grouped_by(temporal_axis=temporal_axis, model=None)

        fig = go.Figure()

        for _, data in data_dict.items():
            models = sorted(set(d["model_name"] for d in data))

            for model in models:
                model_data = [d for d in data if d["model_name"] == model]

                fig.add_trace(
                    go.Bar(
                        x=[d["period"] for d in model_data],
                        y=[d["total_success"] for d in model_data],
                        name=f"{model} – Succès",
                        marker_color="#2ca02c",
                    )
                )

                fig.add_trace(
                    go.Bar(
                        x=[d["period"] for d in model_data],
                        y=[d["total_denials"] for d in model_data],
                        name=f"{model} – Refus",
                        marker_color="#d62728",
                    )
                )

        fig.update_layout(
            title="Répartition des demandes par modèle",
            xaxis_title="Période",
            yaxis_title="Nombre de requêtes",
            barmode="stack",
            template="plotly_white",
        )

        return fig.to_json() if to_json else fig.show()

    def plot_tokens_per_request(self, temporal_axis: str, to_json: bool = True):
        data_dict = self._get_data_grouped_by(temporal_axis=temporal_axis, model=None)

        fig = go.Figure()

        for _, data in data_dict.items():
            for model in sorted(set(d["model_name"] for d in data)):
                model_data = [d for d in data if d["model_name"] == model]

                fig.add_trace(
                    go.Scatter(
                        x=[d["period"] for d in model_data],
                        y=[d["total_tokens"] / d["total_requests"] for d in model_data],
                        mode="lines+markers",
                        name=model,
                    )
                )

        fig.update_layout(
            title="Tokens par requête (efficacité d’usage)",
            xaxis_title="Période",
            yaxis_title="Tokens / requête",
            template="plotly_white",
        )

        return fig.to_json() if to_json else fig.show()

    def plot_environmental_heatmap(self, temporal_axis: str, to_json: bool = True):
        data_dict = self._get_data_grouped_by(temporal_axis=temporal_axis, model=None)

        rows = []
        for _, data in data_dict.items():
            for d in data:
                rows.append(d)

        models = sorted(set(r["model_name"] for r in rows))
        kpis = ["energy_kwh", "gwp_kgCO2eq", "wcf_liters", "pd_mj"]

        z = []
        for model in models:
            model_rows = [r for r in rows if r["model_name"] == model]
            z.append(
                [
                    sum(r[k] for r in model_rows)
                    / sum(r["total_requests"] for r in model_rows)
                    for k in kpis
                ]
            )

        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=kpis,
                y=models,
                colorscale="Reds",
            )
        )

        fig.update_layout(
            title="Profil environnemental par modèle (normalisé par requête)",
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
        }
        self.plot_envir_kpis_over_time(**args, model_name=model_name)
        self.plot_energy_vs_co2(**args)
        self.plot_requests_distribution(**args)
        self.plot_environmental_heatmap(**args)
        self.plot_tokens_per_request(**args)

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
    # just write in your terminal: python -m app.services.plot_manager
    # arguments for kpis_all : temporal_axis, model_name
    # temporal_axis : "W", "M", "Y"
    # model_name : specific model name or None for all model
    pm = PlotManager()
    # print(f"{pm.kpis_all('M', None) = }")
    # print(f"{pm.plot_all('W', None, to_json=False) = }")
    temporal_axis = "W"
    model_name = None
    print(pm.dummy_data(temporal_axis, model_name))
    # pm.plot_envir_kpis_over_time(temporal_axis, model_name, to_json=False)
    pm.plot_all(temporal_axis, model_name, to_json=False)
