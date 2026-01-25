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
        date (date): Date of the record.
    """

    id: int
    nom_modele: str
    total_input_tokens: int
    total_completion_tokens: int
    total_tokens: int
    mean_response_time_ms: float
    total_requests: Optional[int] = None
    total_success: int
    total_denials: Optional[int] = None
    date: date

    model_config = {"from_attributes": True}

    @computed_field
    @property
    def success_rate(self) -> float:
        """
        Calculate the success rate as a percentage.

        Returns:
            float: Success rate (0-100).
        """
        total = self.total_requests or 0
        if total == 0:
            return 0.0
        return (self.total_success / total) * 100

    @computed_field
    @property
    def avg_tokens_per_request(self) -> float:
        """
        Calculate average tokens per request.

        Returns:
            float: Average tokens per request.
        """
        total = self.total_requests or 0
        if total == 0:
            return 0.0
        return self.total_tokens / total
