"""
LLM Metrics Model to track metrics of LLM usage.
"""

from datetime import date

from sqlalchemy import (
    Date,
    Float,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models import Base


class LLMMetrics(Base):
    """
    SQLAlchemy model representing metrics for LLM usage.

    This model stores various metrics related to the usage of large language models (LLMs),
    including token counts, response times, and success indicators.

    Attributes:
        id (int): Primary key, auto-incremented unique identifier.
        nom_modele (str): Name of the LLM model used (max 100 characters). Required.
        total_input_tokens (int): Number of tokens in the input/prompt. Required.
        total_completion_tokens (int): Number of tokens in the completion/response. Required.
        total_tokens (int): Total number of tokens (prompt + completion). Required.
        mean_response_time_ms (float): Time taken to get the response in milliseconds. Required.
        total_success (int): Number of successful LLM calls. Required.
        total_denials (int): Number of denied LLM calls. Optional.
        usage_date (date): Date of the metrics record. Must be unique per model and date. Required.
    """

    __tablename__ = "llm_usage_journalier"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    nom_modele: Mapped[str] = mapped_column(String(100), nullable=False)
    total_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_completion_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    mean_response_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    total_requests: Mapped[int] = mapped_column(Integer, nullable=False)
    total_success: Mapped[int] = mapped_column(Integer, nullable=False)
    total_denials: Mapped[int | None] = mapped_column(Integer, nullable=True)
    usage_date: Mapped[date] = mapped_column(
        Date, default=date.today, unique=True, nullable=False
    )
    __table_args__ = (UniqueConstraint("usage_date", "nom_modele"),)
