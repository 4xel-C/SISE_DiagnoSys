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
        self._cache: dict[tuple[str, list[str] | str], __quelquechose__] = {}
        # ie. dict[number of days, list of models or all] = __quelquechose__

        # check for today's date to manage cache validity
        self._today: date = date.today()

        # KPI units dict
        self._kpi_units_dict: dict[str, str] = {
            "gwp_kgCO2eq": "kgCO2eq",
            "wcf_liters": "l",
            "adpe_mgSbEq": "mgSbEq",
            "energy_kwh": "kwh",
            "total_requests": "nombre de requêtes",
        }

        # comparison_dict import logic
        _fp: str = "data/comparison_dict.json"
        self._comparion_dict_path: str = (
            comparison_dict_path if comparison_dict_path else _fp
        )
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

    def _get_data(
        self, temporal_axis: str, models: list[str] | None = None
    ) -> dict[tuple[str, str | None], list[dict]]:
        cache_key = (temporal_axis, tuple(models) if models else "all")
        if cache_key in self._cache:
            return self._cache[cache_key]

        results = self.llm_usage.get_all_group_by(
            {"temporal_axis": temporal_axis, "model": models}
        )
        self._cache[cache_key] = results
        return results
