import logging
from collections import defaultdict
from datetime import date, timedelta
import bisect
import json
import numpy as np
from plotly import graph_objects as go
from typing import List

# Dummy schema to simulate your Pydantic schema
class LLMMetricsSchema:
    def __init__(
        self, nom_modele: str, total_input_tokens: int, total_completion_tokens: int,
        total_tokens: int, mean_response_time_ms: float, total_requests: int,
        total_success: int, total_denials: int, energy_kwh: float,
        gwp_kgCO2eq: float, adpe_mgSbEq: float, pd_mj: float, wcf_liters: float,
        usage_date: date
    ):
        self.nom_modele = nom_modele
        self.total_input_tokens = total_input_tokens
        self.total_completion_tokens = total_completion_tokens
        self.total_tokens = total_tokens
        self.mean_response_time_ms = mean_response_time_ms
        self.total_requests = total_requests
        self.total_success = total_success
        self.total_denials = total_denials
        self.energy_kwh = energy_kwh
        self.gwp_kgCO2eq = gwp_kgCO2eq
        self.adpe_mgSbEq = adpe_mgSbEq
        self.pd_mj = pd_mj
        self.wcf_liters = wcf_liters
        self.usage_date = usage_date

# Dummy LLMUsageService
class LLMUsageService:
    def get_all(self) -> List[LLMMetricsSchema]:
        return []  # will override in main for fake data

logger = logging.getLogger(__name__)

class PlotManager:
    def __init__(self, llm_usage: LLMUsageService = LLMUsageService(), comparison_dict_path: str | None = None) -> None:
        self.llm_usage = llm_usage

        # cache raw data
        self._cache: dict[tuple[str, str | list[str]], list[LLMMetricsSchema]] = {}

        # cache aggregated KPI
        self._kpi_cache: dict[tuple[str, str | None], dict[str, float]] = {}

        # KPI units
        self._kpi_units_dict: dict[str, str] = {
            "gwp_kgCO2eq": "kgCO2eq",
            "wcf_liters": "l",
            "adpe_mgSbEq": "mgSbEq",
            "energy_kwh": "kwh",
            "total_requests": "nombre de requêtes",
        }

        self._comparison_dict: dict[str, dict[float, str]] = {
            "gwp_kgCO2eq": {100: "une voiture pour 1 km", 500: "un trajet en avion court"},
            "wcf_liters": {1000: "une douche de 10 min", 5000: "bain complet"},
            "energy_kwh": {10: "allumer 10 ampoules pendant 1h", 100: "utiliser un four 1h"}
        }

    # -----------------------------
    # Helper: generate dynamic model palette
    # -----------------------------
    def _get_model_palette(self, models: list[str]) -> dict[str, str]:
        base_colors = [
            "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
            "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
        ]
        palette = {}
        for i, model in enumerate(sorted(models)):
            palette[model] = base_colors[i % len(base_colors)]
        return palette

    # -----------------------------
    # Data aggregation
    # -----------------------------
    def _get_aggregated_metric(self, temporal_axis: str, metric: str, model: str | None = None) -> dict[tuple[str, str], float]:
        if model is None:
            model = "all"

        cache_key = (temporal_axis, model, metric)
        if cache_key in self._cache and self._cache[cache_key]:
            data = self._cache[cache_key]
        else:
            data = self.llm_usage.get_all()
            self._cache[cache_key] = data

        result: dict[tuple[str, str], float] = defaultdict(float)
        for row in data:
            if model not in ("all", row.nom_modele):
                continue
            d = row.usage_date
            if temporal_axis == "W":
                period_key = (d.isocalendar()[0], d.isocalendar()[1])
            elif temporal_axis == "M":
                period_key = (d.year, d.month)
            elif temporal_axis == "Y":
                period_key = (d.year,)
            else:
                raise ValueError("temporal_axis must be one of W, M, Y")

            result[(str(period_key), row.nom_modele)] += getattr(row, metric) or 0.0

        return result

    def _aggregate_kpis(self, temporal_axis: str, model_name: str | None, data: list[LLMMetricsSchema]) -> dict[str, float]:
        cache_key = (temporal_axis, model_name)
        if cache_key in self._kpi_cache:
            return self._kpi_cache[cache_key]

        totals: dict[str, float] = {kpi: 0.0 for kpi in self._kpi_units_dict}
        for row in data:
            for kpi in totals:
                value = getattr(row, kpi)
                if value is not None:
                    totals[kpi] += value

        self._kpi_cache[cache_key] = totals
        return totals

    # -----------------------------
    # KPI Methods
    # -----------------------------
    def make_a_comparison(self, which: str, value: float) -> str:
        kpi_dict = self._comparison_dict.get(which)
        if not kpi_dict:
            raise ValueError(f"comparison dict for {which} is None.")
        thresholds = sorted(kpi_dict.keys())
        idx = bisect.bisect_right(thresholds, value)
        if idx == 0:
            return "Valeur trop faible pour une comparaison."
        lower_key = thresholds[idx - 1]
        ratio = round(value / lower_key, 2)
        return f"Soit {ratio}x {kpi_dict[lower_key]}"

    def _format_kpi_value(self, value: float, unit: str, rounded_to: int = 2) -> str:
        return f"{round(value, rounded_to)}{unit}"

    def get_kpi_statistic(self, which: str, temporal_axis: str, model_name: str | None, data: list[LLMMetricsSchema]) -> dict[str, str]:
        aggregated = self._aggregate_kpis(temporal_axis, model_name, data)
        value = aggregated[which]
        unit = self._kpi_units_dict[which]
        formatted_value = self._format_kpi_value(value, unit)
        comparison = self.make_a_comparison(which, value)
        return {"value": formatted_value, "comparison": comparison}

    # -----------------------------
    # Plot Methods
    # -----------------------------
    def plot_environmental_trend(self, temporal_axis: str = "M") -> str:
        metrics = ["gwp_kgCO2eq", "wcf_liters", "energy_kwh"]
        all_models = set()
        for metric in metrics:
            agg = self._get_aggregated_metric(temporal_axis, metric)
            all_models.update({m for (_, m) in agg.keys()})
        palette = self._get_model_palette(list(all_models))

        fig = go.Figure()
        for metric in metrics:
            agg = self._get_aggregated_metric(temporal_axis, metric)
            for m in all_models:
                model_items = [(dt, val) for (dt, mod), val in agg.items() if mod == m]
                if not model_items:
                    continue
                model_items.sort(key=lambda x: x[0])
                x = [dt for dt, _ in model_items]
                y = [val for _, val in model_items]
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode="lines+markers", name=f"{m} - {metric}",
                    line=dict(color=palette.get(m))
                ))
        fig.update_layout(title="Évolution de l'impact environnemental", xaxis_title="Date", yaxis_title="Valeur", template="plotly_white")
        return fig

    def plot_perf_vs_impact(self) -> str:
        cache_key = ("all", "all")
        data = self._cache.get(cache_key) or self.llm_usage.get_all()
        self._cache[cache_key] = data

        models = {row.nom_modele for row in data}
        palette = self._get_model_palette(list(models))

        fig = go.Figure()
        for row in data:
            y_val = row.gwp_kgCO2eq / row.total_requests if row.total_requests else 0
            fig.add_trace(go.Scatter(
                x=[row.mean_response_time_ms], y=[y_val],
                mode="markers", name=row.nom_modele,
                marker=dict(color=palette.get(row.nom_modele), size=10)
            ))
        fig.update_layout(title="Performance vs Impact par requête", xaxis_title="Mean Response Time (ms)", yaxis_title="GWP per request (kgCO2eq)", template="plotly_white")
        return fig

    def plot_token_distribution(self) -> str:
        cache_key = ("all", "all")
        data = self._cache.get(cache_key) or self.llm_usage.get_all()
        self._cache[cache_key] = data

        totals_input: dict[str, float] = defaultdict(float)
        totals_completion: dict[str, float] = defaultdict(float)
        for row in data:
            totals_input[row.nom_modele] += row.total_input_tokens
            totals_completion[row.nom_modele] += row.total_completion_tokens

        models = list(set(totals_input.keys()) | set(totals_completion.keys()))
        palette = self._get_model_palette(models)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=models, y=[totals_input[m] for m in models], name="Input tokens", marker=dict(color=[palette[m] for m in models])))
        fig.add_trace(go.Bar(x=models, y=[totals_completion[m] for m in models], name="Completion tokens", marker=dict(color=[palette[m] for m in models])))
        fig.update_layout(barmode="stack", title="Répartition des tokens par modèle", xaxis_title="Modèle", yaxis_title="Nombre de tokens", template="plotly_white")
        return fig

    def plot_success_rate(self) -> str:
        cache_key = ("all", "all")
        data = self._cache.get(cache_key) or self.llm_usage.get_all()
        self._cache[cache_key] = data

        rates: dict[str, float] = defaultdict(float)
        counts: dict[str, int] = defaultdict(int)
        for row in data:
            total = row.total_requests or 0
            if total > 0:
                rates[row.nom_modele] += row.total_success / total * 100
                counts[row.nom_modele] += 1

        avg_rates = {m: rates[m] / counts[m] for m in rates}
        palette = self._get_model_palette(list(avg_rates.keys()))

        fig = go.Figure()
        fig.add_trace(go.Bar(x=list(avg_rates.keys()), y=list(avg_rates.values()), name="Success rate (%)", marker=dict(color=[palette[m] for m in avg_rates.keys()])))
        fig.update_layout(title="Taux de succès moyen par modèle", xaxis_title="Modèle", yaxis_title="Success rate (%)", template="plotly_white")
        return fig

    def plot_tokens_per_request_over_time(self, temporal_axis: str = "M") -> str:
        cache_key = ("all", "all")
        data = self._cache.get(cache_key) or self.llm_usage.get_all()
        self._cache[cache_key] = data

        models = {row.nom_modele for row in data}
        palette = self._get_model_palette(list(models))

        fig = go.Figure()
        for m in models:
            model_items = [(row.usage_date, row.total_tokens / row.total_requests if row.total_requests else 0)
                           for row in data if row.nom_modele == m]
            model_items.sort(key=lambda x: x[0])
            x = [dt for dt, _ in model_items]
            y = [val for _, val in model_items]
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=m, line=dict(color=palette[m])))
        fig.update_layout(title="Évolution des tokens par requête", xaxis_title="Date", yaxis_title="Tokens / requête", template="plotly_white")
        return fig

    def plot_success_rate_over_time(self, temporal_axis: str = "M") -> str:
        cache_key = ("all", "all")
        data = self._cache.get(cache_key) or self.llm_usage.get_all()
        self._cache[cache_key] = data

        models = {row.nom_modele for row in data}
        palette = self._get_model_palette(list(models))

        fig = go.Figure()
        for m in models:
            model_items = [(row.usage_date, row.total_success / row.total_requests * 100 if row.total_requests else 0)
                           for row in data if row.nom_modele == m]
            model_items.sort(key=lambda x: x[0])
            x = [dt for dt, _ in model_items]
            y = [val for _, val in model_items]
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=m, line=dict(color=palette[m])))
        fig.update_layout(title="Évolution du taux de succès", xaxis_title="Date", yaxis_title="Success rate (%)", template="plotly_white")
        return fig

    # -----------------------------
    # Facade
    # -----------------------------
    def plot_all(self, temporal_axis: str = "M", model_name: str | None = None) -> dict[str, str]:
        for metric in ["gwp_kgCO2eq", "wcf_liters", "energy_kwh"]:
            self._get_aggregated_metric(temporal_axis, metric, model_name)

        return {
            "environmental_trend": self.plot_environmental_trend(temporal_axis),
            "perf_vs_impact": self.plot_perf_vs_impact(),
            "token_distribution": self.plot_token_distribution(),
            "success_rate": self.plot_success_rate(),
            "tokens_per_request_over_time": self.plot_tokens_per_request_over_time(temporal_axis),
            "success_rate_over_time": self.plot_success_rate_over_time(temporal_axis),
        }

# =========================
# Test with fake data
# =========================
if __name__ == "__main__":
    from random import randint, uniform
    import json

    # Generate fake data
    models = ["small", "medium", "large"]
    fake_data = []
    start_date = date(2024, 1, 1)
    for i in range(30):  # 30 days
        for m in models:
            fake_data.append(
                LLMMetricsSchema(
                    nom_modele=m,
                    total_input_tokens=randint(5000, 20000),
                    total_completion_tokens=randint(2000, 10000),
                    total_tokens=randint(10000, 30000),
                    mean_response_time_ms=uniform(100, 500),
                    total_requests=randint(50, 200),
                    total_success=randint(40, 200),
                    total_denials=randint(0, 10),
                    energy_kwh=uniform(10, 100),
                    gwp_kgCO2eq=uniform(5, 50),
                    adpe_mgSbEq=uniform(0.1, 2),
                    pd_mj=uniform(10, 100),
                    wcf_liters=uniform(50, 500),
                    usage_date=start_date + timedelta(days=i)
                )
            )

    class FakeService(LLMUsageService):
        def get_all(self) -> List[LLMMetricsSchema]:
            return fake_data

    manager = PlotManager(FakeService())

    print("\n=== Plot Example ===")
    plots = manager.plot_all("M")
    for name, fig in plots.items():
        fig.show()