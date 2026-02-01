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

from pydantic import BaseModel, computed_field, field_validator

from app.rag.llm_options import MistralModel


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
        cout_total_usd (float): Total cost in USD.
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
    cout_total_usd: Optional[float] = None
    energy_kwh: Optional[float] = None
    gwp_kgCO2eq: Optional[float] = None
    adpe_mgSbEq: Optional[float] = None
    pd_mj: Optional[float] = None
    wcf_liters: Optional[float] = None
    usage_date: date

    model_config = {"from_attributes": True}

    @field_validator("nom_modele")
    @classmethod
    def validate_nom_modele(cls, value: str) -> str:
        """Validate that nom_modele is a valid MistralModel."""
        valid_models = MistralModel.all_models()
        if value not in valid_models:
            raise ValueError(
                f"Invalid model '{value}'. Must be one of: {', '.join(valid_models)}"
            )
        return value

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


class AggregatedMetricsSchema(BaseModel):
    """
    Pydantic schema for aggregated LLM usage metrics.

    Used for data aggregated by day, month, or year.

    Attributes:
        period (str): The period identifier (e.g., "2024-01-15", "2024-01", "2024").
        nom_modele (str): Model name ("all" if aggregated across all models).
        total_input_tokens (int): Sum of input tokens for the period.
        total_completion_tokens (int): Sum of completion tokens for the period.
        total_tokens (int): Sum of all tokens for the period.
        total_requests (int): Total number of requests for the period.
        total_success (int): Total successful requests for the period.
        total_denials (int): Total denied requests for the period.
        mean_response_time_ms (float): Weighted average response time in ms.
        cout_total_usd (float): Total cost in USD for the period.
        energy_kwh (float): Total energy consumption in kWh.
        gwp_kgCO2eq (float): Total Global Warming Potential in kg CO2 equivalent.
        adpe_mgSbEq (float): Total Abiotic Depletion Potential in mg Sb equivalent.
        pd_mj (float): Total primary energy demand in MJ.
        wcf_liters (float): Total water consumption footprint in liters.
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
    cout_total_usd: float
    energy_kwh: float
    gwp_kgCO2eq: float
    adpe_mgSbEq: float
    pd_mj: float
    wcf_liters: float

    model_config = {"from_attributes": True}

    @field_validator("nom_modele")
    @classmethod
    def validate_nom_modele(cls, value: str) -> str:
        """Validate that nom_modele is a valid MistralModel or 'all'."""
        valid_models = MistralModel.all_models() + ["all"]
        if value not in valid_models:
            raise ValueError(
                f"Invalid model '{value}'. Must be one of: {', '.join(valid_models)}"
            )
        return value

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
    
    def format_for_display(self) -> dict:
        return {
            'energy_kwh': str(round(self.energy_kwh, 2)) + 'kWh',
            'gwp_kgCO2eq': str(round(self.gwp_kgCO2eq, 4)) + 'kg',
            'cout_total_usd': '$' + str(round(self.cout_total_usd, 2)),
            'mean_response_time_ms': str(round(self.mean_response_time_ms)) + 'ms',
            'total_tokens': self.total_tokens
        }

    @staticmethod
    def get_metrics_fields() -> list[str]:
        """
        Get a list of all metric field names.

        Returns:
            list[str]: List of metric field names.
        """
        return [
            "total_input_tokens",
            "total_completion_tokens",
            "total_tokens",
            "mean_response_time_ms",
            "total_requests",
            "total_success",
            "total_denials",
            "cout_total_usd",
            "energy_kwh",
            "gwp_kgCO2eq",
            "adpe_mgSbEq",
            "pd_mj",
            "wcf_liters",
        ]
