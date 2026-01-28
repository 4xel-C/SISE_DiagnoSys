import logging
from collections import defaultdict
from datetime import date

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

        # cache variables
        self._today_metrics: LLMMetricsSchema | None = (
            None  # cache for a single DB call
        )
        self._last_n_days_metrics: dict[int, list[LLMMetricsSchema]] = {}

    ################################################################
    # HELPER METHODS : DATA RETRIEVAL
    ################################################################

    def _get_today_metrics_for_all(self, force_refresh: bool = False):
        """Retrieve today's metrics once and cache them for reuse."""
        if self._today_metrics is None or force_refresh:
            today_metrics = self.llm_usage.get_today()
            # adding every metrics together
            if today_metrics:
                combined_metrics: dict[str, float | int | str | date] = {
                    "id": 0,
                    "nom_modele": "combined",
                    "total_input_tokens": sum(
                        m.total_input_tokens for m in today_metrics
                    ),
                    "total_completion_tokens": sum(
                        m.total_completion_tokens for m in today_metrics
                    ),
                    "total_tokens": sum(m.total_tokens for m in today_metrics),
                    "mean_response_time_ms": sum(
                        m.mean_response_time_ms for m in today_metrics
                    )
                    / len(today_metrics),
                    "total_requests": sum(m.total_requests or 0 for m in today_metrics),
                    "total_success": sum(m.total_success for m in today_metrics),
                    "total_denials": sum(m.total_denials or 0 for m in today_metrics),
                    "gco2": sum(m.gco2 for m in today_metrics),
                    "water_ml": sum(m.water_ml for m in today_metrics),
                    "mgSb": sum(m.mgSb for m in today_metrics),
                    "usage_date": today_metrics[0].usage_date,
                }
                self._today_metrics = LLMMetricsSchema.model_validate(combined_metrics)
            else:
                self._today_metrics = None

        return self._today_metrics

    def _get_last_n_days(self, n: int) -> list[LLMMetricsSchema]:
        """Retrieve LLM metrics for the last n days.

        Args:
            n (int): Number of days to retrieve metrics for.

        Returns:
            list[LLMMetricsSchema]: List of metrics for the last n days.
        """
        if n not in self._last_n_days_metrics:
            metrics = self.llm_usage.get_last_n_days(n)
            self._last_n_days_metrics[n] = metrics
        return self._last_n_days_metrics[n]

    ################################################################
    # KPI METHODS
    ################################################################

    def request_kpi_today(
        self, kpi: str, comparisons: bool = False
    ) -> tuple[float, str, str]:
        """Retrieve specific KPI for today's LLM requests.

        Args:
            kpi (str): KPI to retrieve from: CO2, water, antimony, total_requests.
            comparisons (bool, optional): Whether to include comparison phrasing. Defaults to False.

        Returns:
            tuple[float, str, str]: KPI value, unit, and comparison phrasing if requested.
        """
        metrics = self._get_today_metrics_for_all()
        if not metrics:
            return 0.0, "", ""
        kpi_map: dict[str, tuple[float, str]] = {
            "CO2": (metrics.gco2, "gCO2"),
            "water": (metrics.water_ml, "ml"),
            "antimony": (metrics.mgSb, "mg"),
            "total_requests": (metrics.total_requests or 0, "requests"),
        }

        if kpi not in kpi_map:
            raise ValueError(f"Invalid KPI requested: {kpi}")

        value, unit = kpi_map[kpi]
        comparison_text = ""
        if comparisons:
            # Placeholder for actual comparison logic
            # TODO: implement comparison logic
            pass

        return value, unit, comparison_text

    def make_a_comparison(self, kpi: str) -> str:
        """Create a comparison string for a given KPI over a specified period.

        Args:
            kpi (str): KPI to compare: CO2, water, antimony, total_requests.

        Returns:
            str: Comparison string.
        """
        match kpi:
            case "CO2":
                ...
            case "water":
                ...
            case "antimony":
                ...
            case "total_requests":
                ...
            case _:
                raise ValueError(f"Invalid KPI for comparison: {kpi}")
        return ""

    ################################################################
    # PLOTTING METHODS
    ################################################################

    def plot_total_request_per_day(
        self, days: int = 7, to_json: bool = False
    ) -> go.Figure:
        """Generate a plot of total LLM requests per day over the last n days.

        Args:
            days (int, optional): Number of days to include in the plot. Defaults to 7.

        Returns:
            go.Figure: Plotly figure object representing the plot.
        """
        metrics = self._get_last_n_days(days)
        if not metrics:
            logger.warning("No metrics available for plotting.")
            return go.Figure()

        # Aggregate total requests per day
        requests_per_day = defaultdict(int)
        for metric in metrics:
            requests_per_day[metric.usage_date] += metric.total_requests or 0

        # Prepare data for plotting
        dates = sorted(requests_per_day.keys())
        total_requests = [requests_per_day[date] for date in dates]

        # Create Plotly figure
        fig = go.Figure(data=go.Bar(x=dates, y=total_requests))
        fig.update_layout(
            title=f"Total LLM Requests per Day (Last {days} Days)",
            xaxis_title="Date",
            yaxis_title="Total Requests",
        )

        if to_json:
            return fig.to_json()
        return fig

    def plot_tokens_per_day(self, days: int = 7, to_json: bool = False) -> go.Figure:
        """Generate a plot of total tokens used per day over the last n days.

        Args:
            days (int, optional): Number of days to include in the plot. Defaults to 7.

        Returns:
            go.Figure: Plotly figure object representing the plot.
        """
        metrics = self._get_last_n_days(days)
        if not metrics:
            logger.warning("No metrics available for plotting.")
            return go.Figure()

        # Aggregate total tokens per day
        tokens_per_day = defaultdict(int)
        for metric in metrics:
            tokens_per_day[metric.usage_date] += metric.total_tokens

        # Prepare data for plotting
        dates = sorted(tokens_per_day.keys())
        total_tokens = [tokens_per_day[date] for date in dates]

        # Create Plotly figure
        fig = go.Figure(data=go.Scatter(x=dates, y=total_tokens, mode="lines+markers"))
        fig.update_layout(
            title=f"Total Tokens Used per Day (Last {days} Days)",
            xaxis_title="Date",
            yaxis_title="Total Tokens",
        )
        if to_json:
            return fig.to_json()
        return fig

    def plot_success_rate_per_day(
        self, days: int = 7, to_json: bool = False
    ) -> go.Figure:
        """Generate a plot of success rate per day over the last n days.

        Args:
            days (int, optional): Number of days to include in the plot. Defaults to 7.

        Returns:
            go.Figure: Plotly figure object representing the plot.
        """
        metrics = self._get_last_n_days(days)
        if not metrics:
            logger.warning("No metrics available for plotting.")
            return go.Figure()

        # Aggregate success rate per day
        success_rate_per_day = defaultdict(list)
        for metric in metrics:
            success_rate_per_day[metric.usage_date].append(metric.success_rate)

        # Calculate average success rate per day
        avg_success_rate_per_day = {
            date: sum(rates) / len(rates)
            for date, rates in success_rate_per_day.items()
        }

        # Prepare data for plotting
        dates = sorted(avg_success_rate_per_day.keys())
        success_rates = [avg_success_rate_per_day[date] for date in dates]

        # Create Plotly figure
        fig = go.Figure(data=go.Scatter(x=dates, y=success_rates, mode="lines+markers"))
        fig.update_layout(
            title=f"Average Success Rate per Day (Last {days} Days)",
            xaxis_title="Date",
            yaxis_title="Success Rate (%)",
        )
        if to_json:
            return fig.to_json()
        return fig

    def plot_co2_emissions_per_day(
        self, days: int = 7, to_json: bool = False
    ) -> go.Figure:
        """Generate a plot of CO2 emissions per day over the last n days.

        Args:
            days (int, optional): Number of days to include in the plot. Defaults to 7.

        Returns:
            go.Figure: Plotly figure object representing the plot.
        """
        metrics = self._get_last_n_days(days)
        if not metrics:
            logger.warning("No metrics available for plotting.")
            return go.Figure()

        # Aggregate CO2 emissions per day
        co2_per_day = defaultdict(float)
        for metric in metrics:
            co2_per_day[metric.usage_date] += metric.gco2

        # Prepare data for plotting
        dates = sorted(co2_per_day.keys())
        total_co2 = [co2_per_day[date] for date in dates]

        # Create Plotly figure
        fig = go.Figure(data=go.Bar(x=dates, y=total_co2))
        fig.update_layout(
            title=f"CO2 Emissions per Day (Last {days} Days)",
            xaxis_title="Date",
            yaxis_title="CO2 Emissions (gCO2)",
        )
        if to_json:
            return fig.to_json()
        return fig

    def plot_reponse_time_per_request(
        self, days: int = 7, to_json: bool = False
    ) -> go.Figure:
        """Generate a plot of average response time per request over the last n days.

        Args:
            days (int, optional): Number of days to include in the plot. Defaults to 7.

        Returns:
            go.Figure: Plotly figure object representing the plot.
        """
        metrics = self._get_last_n_days(days)
        if not metrics:
            logger.warning("No metrics available for plotting.")
            return go.Figure()

        # Aggregate response time per day
        response_time_per_day = defaultdict(list)
        for metric in metrics:
            response_time_per_day[metric.usage_date].append(
                metric.mean_response_time_ms
            )

        # Calculate average response time per day
        avg_response_time_per_day = {
            date: sum(times) / len(times)
            for date, times in response_time_per_day.items()
        }

        # Prepare data for plotting
        dates = sorted(avg_response_time_per_day.keys())
        response_times = [avg_response_time_per_day[date] for date in dates]

        # Create Plotly figure
        fig = go.Figure(
            data=go.Scatter(x=dates, y=response_times, mode="lines+markers")
        )
        fig.update_layout(title=f"Average Response Time per Request (Last {days} Days)")
        # axis:
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Response Time (ms)",
        )
        if to_json:
            return fig.to_json()
        return fig


if __name__ == "__main__":
    # to start : python -m app.services.plot_manager

    # --- Fake service pour tests locaux ---
    class FakeLLMUsageService:
        def get_last_n_days(self, n: int):
            return sample_metrics

        def get_today(self):
            today = max(m.usage_date for m in sample_metrics)
            return [m for m in sample_metrics if m.usage_date == today]

    # --- Instantiate PlotManager with fake service ---
    fake_service = FakeLLMUsageService()
    plot_manager = PlotManager(llm_usage=fake_service)

    # --- Generate and show plots ---
    fig_requests = plot_manager.plot_total_request_per_day(days=7)
    fig_requests.show()

    fig_tokens = plot_manager.plot_tokens_per_day(days=7)
    fig_tokens.show()

    fig_success = plot_manager.plot_success_rate_per_day(days=7)
    fig_success.show()

    fig_co2 = plot_manager.plot_co2_emissions_per_day(days=7)
    fig_co2.show()
