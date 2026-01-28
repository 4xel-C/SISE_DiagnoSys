import logging
from collections import defaultdict
from datetime import date

import numpy as np
from plotly import graph_objects as go

from app.schemas.llm_metrics_schema import LLMMetricsSchema
from app.services.llm_usage_service import LLMUsageService

logger = logging.getLogger(__name__)

sample_metrics: list[LLMMetricsSchema] = [
    LLMMetricsSchema(
        id=1,
        nom_modele="mistral-small",
        total_input_tokens=120_000,
        total_completion_tokens=45_000,
        total_tokens=165_000,
        mean_response_time_ms=210.5,
        total_requests=320,
        total_success=310,
        total_denials=10,
        gco2=185.4,
        water_ml=920.0,
        mgSb=0.82,
        usage_date=date(2024, 11, 1),
    ),
    LLMMetricsSchema(
        id=2,
        nom_modele="mistral-medium",
        total_input_tokens=95_000,
        total_completion_tokens=60_000,
        total_tokens=155_000,
        mean_response_time_ms=340.2,
        total_requests=280,
        total_success=275,
        total_denials=5,
        gco2=240.1,
        water_ml=1_150.0,
        mgSb=1.10,
        usage_date=date(2024, 11, 1),
    ),
    LLMMetricsSchema(
        id=3,
        nom_modele="mistral-small",
        total_input_tokens=140_000,
        total_completion_tokens=52_000,
        total_tokens=192_000,
        mean_response_time_ms=205.8,
        total_requests=360,
        total_success=355,
        total_denials=5,
        gco2=210.6,
        water_ml=1_030.0,
        mgSb=0.91,
        usage_date=date(2024, 11, 2),
    ),
    LLMMetricsSchema(
        id=4,
        nom_modele="mistral-medium",
        total_input_tokens=110_000,
        total_completion_tokens=72_000,
        total_tokens=182_000,
        mean_response_time_ms=355.4,
        total_requests=300,
        total_success=290,
        total_denials=10,
        gco2=265.9,
        water_ml=1_280.0,
        mgSb=1.22,
        usage_date=date(2024, 11, 2),
    ),
    LLMMetricsSchema(
        id=5,
        nom_modele="mistral-large",
        total_input_tokens=80_000,
        total_completion_tokens=95_000,
        total_tokens=175_000,
        mean_response_time_ms=510.7,
        total_requests=150,
        total_success=145,
        total_denials=5,
        gco2=390.3,
        water_ml=1_900.0,
        mgSb=2.05,
        usage_date=date(2024, 11, 2),
    ),
]


class PlotManager:
    def __init__(self, llm_usage: LLMUsageService = LLMUsageService()) -> None:
        """Initialize the PlotManager..

        Args:
            usage_service (LLMUsageService): LLM usage service instance.
        """
        self.llm_usage = llm_usage

        # dict[number of days, list of models or all] = list of schemas
        self._cache: dict[tuple[str, list[str] | str], list[LLMMetricsSchema]] = {}

    ################################################################
    # HELPER METHODS : DATA RETRIEVAL
    ################################################################

    def _get_data_for(
        self, temporal_axis: str, model: str | None
    ) -> list[LLMMetricsSchema]:
        if model is None:
            model = "all"
        # to not overload the database : we cache the data retrieved.
        already_cached = self._cache[(temporal_axis, model)]
        if already_cached:
            return already_cached
        # if not, we get the data with the llm_usage_service

        # TODO: implement the retrieval of all data for number_of_days days and list_of_models models

        data_to_return: list[LLMMetricsSchema] | None = None

        return data_to_return

    ################################################################
    # KPI GETTERS
    ################################################################

    def get_kpi_statistic(self, which: str, add_a_comparison: bool = False):
        if which not in ["CO2", "water", "antimony", "total_requests", "all"]:
            raise ValueError("")  # TODO: implement the error and stuff
        pass

    # ...

    # dummy
    def plot_dummy(self):
        fig = go.Figure()
        x = np.linspace(0, 10, 100)
        y = x  # y = x
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="y = x"))

        # Ajouter des titres
        fig.update_layout(
            title="Graphique de y = x",
            xaxis_title="x",
            yaxis_title="y",
            template="plotly_white",
        )

        # Afficher le graphique
        return fig.to_json()

    def plot_all(self, temporal_axis: str, model_name: str | None = None):
        data = self._get_data_for(temporal_axis=temporal_axis, model=model_name)

        # now that we have data, we dispatch the data to the different plot methods
        plots: dict[str, str] | None = None
        # plots:  {name_of_plot: __json_string__}
        # -> dict of plots

        return plots

    def kpis_all(self, temporal_axis: str, model_name: str | None = None):
        data = self._get_data_for(temporal_axis=temporal_axis, model=model_name)

        # now that we have data, we dispatch the data to the different kpis methods
        kpis: dict[str, dict[float, str]] | None = None
        # kpis : {name_of_kpi: {kpi_value : __value__, kpi_commentary: __commentary__}, ...}
        # -> dict of kpis with their values and their commentary

        return kpis
