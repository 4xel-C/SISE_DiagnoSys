"""
Plot Service Module.

Generates Plotly.js chart data from LLM usage metrics.
"""

import logging
from typing import List, Union, Optional

from app.rag.llm_options import MistralModel
from app.schemas import AggregatedMetricsSchema
from app.services.llm_usage_service import AggTime, LLMUsageService

logger = logging.getLogger(__name__)


class PlotService:
    """Service for generating Plotly.js chart data."""

    def __init__(self, llm_usage_service: LLMUsageService = LLMUsageService()):
        self.llm_usage_service = llm_usage_service

    def line_plot(
        self,
        metric: str,
        agg_time: Union[str, AggTime] = AggTime.DAILY,
        models: List[str] = ["all"],
    ) -> dict:
        """
        Generate a line plot with one line per model.

        Args:
            metric: Field name to plot (e.g: "total_tokens", "total_requests", "energy_kwh", "gwp_kgCO2eq", "adpe_mgSbEq", "wcf_liters", "mean_response_time_ms").
            agg_time: Aggregation time (daily, monthly, yearly).
            models: Optional list of models to filter by.
        Returns:
            dict: Plotly.js structure {"data": [...], "layout": {...}}
        """

        # validate metric
        if metric not in AggregatedMetricsSchema.get_metrics_fields():
            raise ValueError(f"Invalid metric: {metric}")

        for model in models:
            if model != "all" and model not in MistralModel.all_models():
                raise ValueError(f"Invalid model: {model}")

        metrics = self.llm_usage_service.get_aggregated_data(
            agg_time=agg_time, models=models
        )

        # Group by model
        by_model: dict[str, list] = {}

        for m in metrics:
            if m.nom_modele not in by_model:
                by_model[m.nom_modele] = []
            by_model[m.nom_modele].append(m)

        # Sort by period (ascending)
        for records in by_model.values():
            records.sort(key=lambda x: x.period)

        # Build traces
        traces = []
        for model_name, records in by_model.items():
            traces.append(
                {
                    "x": [r.period for r in records],
                    "y": [getattr(r, metric) for r in records],
                    "type": "scatter",
                    "mode": "lines+markers",
                    "name": model_name,
                }
            )

        return {
            "data": traces,
            "layout": {
                "title": metric,
                "autosize": True,
                "xaxis": {
                    "title": agg_time.value
                    if isinstance(agg_time, AggTime)
                    else agg_time
                },
                "yaxis": {"title": metric},
            },
        }

    def pie_plot(
        self,
        metric: str,
        agg_time: Union[str, AggTime] = AggTime.DAILY,
        models: List[str] = MistralModel.all_models(),
    ) -> dict:
        """
        Generate a pie chart showing the distribution of a metric across models.

        Args:
            metric: Field name to plot (e.g: "total_tokens", "total_requests", "energy_kwh", "gwp_kgCO2eq", "adpe_mgSbEq", "wcf_liters", "mean_response_time_ms").
            agg_time: Aggregation time (daily, monthly, yearly).
            models: Optional list of models to filter by.
        Returns:
            dict: Plotly.js structure {"data": [...], "layout": {...}}
        """

        # validate metric
        if metric not in AggregatedMetricsSchema.get_metrics_fields():
            raise ValueError(f"Invalid metric: {metric}")

        metrics = self.llm_usage_service.get_aggregated_data(
            agg_time=agg_time, models=models
        )

        # Aggregate by model
        aggregation: dict[str, float] = {}
        for m in metrics:
            if m.nom_modele not in aggregation:
                aggregation[m.nom_modele] = 0.0
            aggregation[m.nom_modele] += getattr(m, metric)

        # Build pie chart data
        labels = list(aggregation.keys())
        values = [aggregation[label] for label in labels]

        return {
            "data": [
                {
                    "labels": labels,
                    "values": values,
                    "type": "pie",
                }
            ],
            "layout": {
                "title": f"Distribution of {metric}",
                "autosize": True,
            },
        }
    
    def kpis(
        self,
        agg_time: Union[str, AggTime] = AggTime.DAILY,
    ) -> Optional[AggregatedMetricsSchema]:
        """
        Wrapper method to get KPIs.

        Args:
            agg_time: Aggregation time (daily, monthly, yearly).
                       Accepts string or AggTime enum.
            models: Optional list of models to filter by.
                    Accepts strings or MistralModel enums.

        Returns:
            List[AggregatedMetricsSchema]: Aggregated metrics per period and model.
        """
        return self.llm_usage_service.get_current_period_metrics(agg_time)

    def get_metrics_name(self) -> list[str]:
        """
        Wrapper method to get the list of possible metric names.

        Returns:
            list[str]: List of metric field names.
        """
        return AggregatedMetricsSchema.get_metrics_fields()

    def get_possible_agg_times(self) -> list[str]:
        """
        Get the list of possible aggregation times.

        Returns:
            list[str]: List of aggregation time names.
        """
        return [time.value for time in AggTime]

    def get_possible_models(self) -> list[str]:
        """
        Wraper method to get the list of possible model names from LLM usage data.

        Returns:
            list[str]: List of model names.
        """
        return self.llm_usage_service.get_unique_models()
