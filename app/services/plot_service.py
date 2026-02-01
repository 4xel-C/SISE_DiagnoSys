"""
Plot Service Module.

Generates Plotly.js chart data from LLM usage metrics.
"""

import logging
from typing import Union

from app.schemas import AggregatedMetrics
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
    ) -> dict:
        """
        Generate a line plot with one line per model.

        Args:
            metric: Field name to plot (e.g., "tokens", "gco2", "requests").
            agg_time: Aggregation time (daily, monthly, yearly).

        Returns:
            dict: Plotly.js structure {"data": [...], "layout": {...}}
        """

        # validate metric
        if metric not in AggregatedMetrics.get_metrics_name():
            raise ValueError(f"Invalid metric: {metric}")

        metrics = self.llm_usage_service.get_aggregated_data(agg_time=agg_time)

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
                "xaxis": {
                    "title": agg_time.value
                    if isinstance(agg_time, AggTime)
                    else agg_time
                },
                "yaxis": {"title": metric},
            },
        }

    def get_metrics_name(self) -> list[str]:
        """
        Get the list of available metric names.

        Returns:
            list[str]: List of metric field names.
        """
        return AggregatedMetrics.get_metrics_name()

    def get_possible_agg_times(self) -> list[str]:
        """
        Get the list of possible aggregation times.

        Returns:
            list[str]: List of aggregation time names.
        """
        return [time.value for time in AggTime]
