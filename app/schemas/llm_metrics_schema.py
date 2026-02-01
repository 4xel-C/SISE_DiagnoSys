"""
LLM Metrics Schema Module.

This module defines the Pydantic schema for LLMMetrics ORM model.

Example:
    >>> from app.models.llm_usage import LLMMetrics
    >>> from app.schemas import LLMMetricsSchema
    >>> record = session.query(LLMMetrics).first()
    >>> schema = LLMMetricsSchema.model_validate(record)
"""

from datetime import date
from typing import Optional

from pydantic import BaseModel, computed_field


class AggregatedMetrics(BaseModel):
    """
    Aggregated metrics for a period.

    Used to return metrics aggregated by day, month, or year.

    Attributes:
        period (str): The period identifier (e.g., "2024-01-15", "2024-01", "2024").
        nom_modele (str | None): Model name, None if aggregated across all models.
        total_input_tokens (int): Sum of input tokens for the period.
        total_completion_tokens (int): Sum of completion tokens for the period.
        total_tokens (int): Sum of all tokens for the period.
        total_requests (int): Total number of requests for the period.
        total_success (int): Total successful requests for the period.
        total_denials (int): Total denied requests for the period.
        mean_response_time_ms (float): Weighted average response time in ms.
        gco2 (float): Total CO2 emissions in grams.
        water_ml (float): Total water usage in milliliters.
        mgSb (float): Total antimony usage in milligrams.
    """

    period: str
    nom_modele: str
    total_input_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_requests: int
    total_success: int
    total_denials: int
    mean_response_time_ms: float
    gco2: float
    water_ml: float
    mgSb: float

    model_config = {"from_attributes": True}

    @staticmethod
    def get_metrics_name() -> list[str]:
        """
        Get the list of model field names.

        Returns:
            list[str]: List of field names.
        """
        return [
            "input_tokens",
            "completion_tokens",
            "tokens",
            "requests",
            "success",
            "denials",
            "mean_response_time_ms",
            "gco2",
            "water_ml",
            "mgSb",
        ]


class LLMMetricsSchema(BaseModel):
    """
    Pydantic schema for LLM usage metrics.

    Attributes:
        id (int): Primary key.
        nom_modele (str): Name of the LLM model.
        total_input_tokens (int): Total input tokens.
        total_completion_tokens (int): Total completion tokens.
        total_tokens (int): Total tokens (input + completion).
        mean_response_time_ms (float): Average response time in ms.
        total_requests (int): Total number of requests.
        total_success (int): Number of successful requests.
        total_denials (int): Number of denied requests.
        usage_date (date): Date of the record.
        energy_kwh (float): Energy consumption in kWh.
        gwp_kgCO2eq (float): Global Warming Potential in kg CO2 equivalent.
        adpe_mgSbEq (float): Abiotic Depletion Potential (elements) in mg Sb equivalent.
        pd_mj (float): Primary energy demand in MJ.
        wcf_liters (float): Water consumption footprint in liters.
    """

    id: int
    nom_modele: str
    total_input_tokens: int
    total_completion_tokens: int
    total_tokens: int
    mean_response_time_ms: float
    total_requests: int
    total_success: int
    total_denials: Optional[int] = None
    energy_kwh: Optional[float] = None
    gwp_kgCO2eq: Optional[float] = None
    adpe_mgSbEq: Optional[float] = None
    pd_mj: Optional[float] = None
    wcf_liters: Optional[float] = None
    usage_date: date

    model_config = {"from_attributes": True}

    @computed_field
    @property
    def success_rate(self) -> float:
        """
        Calculate the success rate as a percentage.

        Returns:
            float: Success rate (0-100).
        """
        if self.total_requests == 0:
            return 0.0
        return (self.total_success / self.total_requests) * 100

    @computed_field
    @property
    def avg_tokens_per_request(self) -> float:
        """
        Calculate average tokens per request.

        Returns:
            float: Average tokens per request.
        """
        if self.total_requests == 0:
            return 0.0
        return self.total_tokens / self.total_requests
