"""
Plot Service Module.

Generates Plotly.js chart data from LLM usage metrics.
"""

import logging
from typing import Union

from app.services.llm_usage_service import AggLevel, LLMUsageService

logger = logging.getLogger(__name__)


class PlotService:
    """Service for generating Plotly.js chart data."""

    def __init__(self, llm_usage_service: LLMUsageService = LLMUsageService()):
        self.llm_usage_service = llm_usage_service

    def line_plot(
        self,
        metric: str,
        agg_level: Union[str, AggLevel] = AggLevel.DAILY,
    ) -> dict:
        """
        Generate a line plot with one line per model.

        Args:
            metric: Field name to plot (e.g., "total_tokens", "gco2", "total_requests").
            agg_level: Aggregation level (daily, monthly, yearly).

        Returns:
            dict: Plotly.js structure {"data": [...], "layout": {...}}
        """
        metrics = self.llm_usage_service.get_metrics(agg_level=agg_level)

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
                    "title": agg_level.value
                    if isinstance(agg_level, AggLevel)
                    else agg_level
                },
                "yaxis": {"title": metric},
            },
        }
