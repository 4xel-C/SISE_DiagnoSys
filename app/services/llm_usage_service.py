"""
LLM Usage Service Module.

This module provides database operations for tracking LLM usage metrics.
It handles daily aggregation of token usage, costs, and performance metrics.

Example:
    >>> from app.services import LLMUsageService
    >>> service = LLMUsageService()
    >>> service.record_usage("mistral-small", input_tokens=100, output_tokens=50, response_time_ms=200)
"""

import logging
from datetime import date, timedelta

from app.config import Database, db
from app.models import LLMMetrics
from app.rag import LLMUsage
from app.schemas import LLMMetricsSchema

logger = logging.getLogger(__name__)


class LLMUsageService:
    """
    Service class for LLM usage metrics operations.

    Provides methods to record, retrieve, and analyze LLM usage data
    with daily aggregation.

    Example:
        >>> service = LLMUsageService()
        >>> service.record_usage("mistral-small", 100, 50, 200.0)
        >>> today_stats = service.get_today()
    """

    def __init__(self, db_manager: Database = db):
        """
        Initialize the LLMUsageService.

        Args:
            db_manager (Database): Database manager instance. Defaults to global db.
        """
        self.db_manager = db_manager

    ################################################################
    # GET METHODS
    ################################################################

    def get_all(self) -> list[LLMMetricsSchema]:
        """
        Retrieve all LLM usage records.

        Returns:
            list[LLMMetricsSchema]: List of all usage records.

        Example:
            >>> all_records = service.get_all()
        """
        logger.debug("Fetching all LLM usage records.")
        result = []
        with self.db_manager.session() as session:
            records = (
                session.query(LLMMetrics).order_by(LLMMetrics.usage_date.desc()).all()
            )
            logger.debug(f"Found {len(records)} LLM usage records.")
            for record in records:
                result.append(LLMMetricsSchema.model_validate(record))
        return result

    def get_by_date(self, target_date: date) -> list[LLMMetricsSchema]:
        """
        Retrieve usage records for a specific date.

        Args:
            target_date (date): The date to query.

        Returns:
            list[LLMMetricsSchema]: List of records for that date (one per model).

        Example:
            >>> from datetime import date
            >>> records = service.get_by_date(date(2024, 1, 15))
        """
        logger.debug(f"Fetching LLM usage for date={target_date}.")
        result = []

        with self.db_manager.session() as session:
            records = (
                session.query(LLMMetrics)
                .filter(LLMMetrics.usage_date == target_date)
                .all()
            )
            logger.debug(f"Found {len(records)} records for {target_date}.")

            for record in records:
                result.append(LLMMetricsSchema.model_validate(record))

        return result

    def get_today(self) -> list[LLMMetricsSchema]:
        """
        Retrieve today's usage records.

        Returns:
            list[LLMMetricsSchema]: List of today's records (one per model).

        Example:
            >>> today = service.get_today()
        """
        return self.get_by_date(date.today())

    def get_by_model(self, model_name: str) -> list[LLMMetricsSchema]:
        """
        Retrieve all usage records for a specific model.

        Args:
            model_name (str): Name of the model (e.g., "mistral-small").

        Returns:
            list[LLMMetricsSchema]: List of records for that model.

        Example:
            >>> mistral_usage = service.get_by_model("mistral-small")
        """
        logger.debug(f"Fetching LLM usage for model={model_name}.")
        result = []
        with self.db_manager.session() as session:
            records = (
                session.query(LLMMetrics)
                .filter(LLMMetrics.nom_modele == model_name)
                .order_by(LLMMetrics.usage_date.desc())
                .all()
            )
            logger.debug(f"Found {len(records)} records for model {model_name}.")
            for record in records:
                result.append(LLMMetricsSchema.model_validate(record))
        return result

    def get_by_date_range(
        self, start_date: date, end_date: date
    ) -> list[LLMMetricsSchema]:
        """
        Retrieve usage records within a date range.

        Args:
            start_date (date): Start of the range (inclusive).
            end_date (date): End of the range (inclusive).

        Returns:
            list[LLMMetricsSchema]: List of records in the range.

        Example:
            >>> from datetime import date, timedelta
            >>> week_ago = date.today() - timedelta(days=7)
            >>> records = service.get_by_date_range(week_ago, date.today())
        """
        logger.debug(f"Fetching LLM usage from {start_date} to {end_date}.")
        result = []
        with self.db_manager.session() as session:
            records = (
                session.query(LLMMetrics)
                .filter(
                    LLMMetrics.usage_date >= start_date,
                    LLMMetrics.usage_date <= end_date,
                )
                .order_by(LLMMetrics.usage_date.desc())
                .all()
            )
            logger.debug(f"Found {len(records)} records in date range.")
            for record in records:
                result.append(LLMMetricsSchema.model_validate(record))
        return result

    def get_last_n_days(self, n: int = 7) -> list[LLMMetricsSchema]:
        """
        Retrieve usage records for the last N days.

        Args:
            n (int): Number of days to look back. Defaults to 7.

        Returns:
            list[LLMMetricsSchema]: List of records for the last N days.

        Example:
            >>> last_week = service.get_last_n_days(7)
        """
        start_date = date.today() - timedelta(days=n - 1)
        return self.get_by_date_range(start_date, date.today())

    ################################################################
    # RECORD / UPDATE METHODS
    ################################################################

    def record_usage(
        self,
        model_name: str,
        usage: LLMUsage,
        success: bool = True,
    ) -> LLMMetricsSchema:
        """
        Record a new LLM usage. Updates today's record if exists, creates one otherwise.

        Args:
            model (MistralModel): The model used.
            usage (LLMUsage): Usage statistics including token counts and latency.
            success (bool): Whether the request was successful. Defaults to True.

        Returns:
            LLMMetricsSchema: The updated or created record.

        Example:
            >>> record = service.record_usage(
            ...     MistralModel.MISTRAL_SMALL,
            ...     usage=LLMUsage(input_tokens=150, output_tokens=75, latency_ms=250.5),
            ...     success=True
            ... )
        """

        logger.debug(f"Recording LLM usage for model={model_name}.")
        today = date.today()

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        response_time_ms = usage.latency_ms
        energy_kwh = usage.energy_kwh
        gwp_kgCO2eq = usage.gwp_kgCO2eq
        adpe_mgSbEq = usage.adpe_mgSbEq
        pd_mj = usage.pd_mj
        wcf_liters = usage.wcf_liters

        # Connect to the db and record usage
        with self.db_manager.session() as session:
            # Check if record exists for today and this model
            record = (
                session.query(LLMMetrics)
                .filter(
                    LLMMetrics.usage_date == today, LLMMetrics.nom_modele == model_name
                )
                .first()
            )

            # Update if a record exists (update daily metrics)
            if record:
                # Update existing record
                record.total_input_tokens += input_tokens
                record.total_completion_tokens += output_tokens
                record.total_tokens += input_tokens + output_tokens
                record.total_requests += 1
                record.energy_kwh = (record.energy_kwh or 0) + (energy_kwh or 0)
                record.gwp_kgCO2eq = (record.gwp_kgCO2eq or 0) + (gwp_kgCO2eq or 0)
                record.adpe_mgSbEq = (record.adpe_mgSbEq or 0) + (adpe_mgSbEq or 0)
                record.pd_mj = (record.pd_mj or 0) + (pd_mj or 0)
                record.wcf_liters = (record.wcf_liters or 0) + (wcf_liters or 0)

                # Update mean response time (running average)
                total_requests = record.total_requests
                old_mean = record.mean_response_time_ms
                record.mean_response_time_ms = (
                    old_mean * (total_requests - 1) + response_time_ms
                ) / total_requests

                if success:
                    record.total_success += 1
                else:
                    record.total_denials = (record.total_denials or 0) + 1

                logger.info(f"Updated LLM usage for {model_name} on {today}.")

            # If no record exists for the day, create a new one.
            else:
                # Create new record
                record = LLMMetrics(
                    nom_modele=model_name,
                    total_input_tokens=input_tokens,
                    total_completion_tokens=output_tokens,
                    total_tokens=input_tokens + output_tokens,
                    mean_response_time_ms=response_time_ms,
                    total_requests=1,
                    total_success=1 if success else 0,
                    total_denials=0 if success else 1,
                    energy_kwh=energy_kwh,
                    gwp_kgCO2eq=gwp_kgCO2eq,
                    adpe_mgSbEq=adpe_mgSbEq,
                    pd_mj=pd_mj,
                    wcf_liters=wcf_liters,
                    usage_date=today,
                )
                session.add(record)
                logger.info(
                    f"Created new LLM usage record for {model_name} on {today}."
                )

            session.commit()
            return LLMMetricsSchema.model_validate(record)

    ################################################################
    # AGGREGATE / STATS METHODS
    ################################################################

    def get_total_tokens_today(self) -> int:
        """
        Get total tokens used today across all models.

        Returns:
            int: Total tokens used today.

        Example:
            >>> total = service.get_total_tokens_today()
        """
        records = self.get_today()
        return sum(r.total_tokens for r in records)

    def get_total_requests_today(self) -> int:
        """
        Get total requests made today across all models.

        Returns:
            int: Total requests today.

        Example:
            >>> requests = service.get_total_requests_today()
        """
        records = self.get_today()
        return sum(r.total_requests or 0 for r in records)

    def get_summary(self) -> dict:
        """
        Get a summary of all-time LLM usage.

        Returns:
            dict: Summary with total tokens, requests, and per-model breakdown.

        Example:
            >>> summary = service.get_summary()
            >>> print(summary["total_tokens"])
        """
        all_records = self.get_all()

        total_input = sum(r.total_input_tokens for r in all_records)
        total_output = sum(r.total_completion_tokens for r in all_records)
        total_tokens = sum(r.total_tokens for r in all_records)
        total_requests = sum(r.total_requests or 0 for r in all_records)
        total_success = sum(r.total_success for r in all_records)
        total_denials = sum(r.total_denials or 0 for r in all_records)

        # Per-model breakdown
        models: dict[str, dict] = {}
        for record in all_records:
            if record.nom_modele not in models:
                models[record.nom_modele] = {
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_tokens": 0,
                    "total_requests": 0,
                }
            models[record.nom_modele]["total_input_tokens"] += record.total_input_tokens
            models[record.nom_modele]["total_output_tokens"] += (
                record.total_completion_tokens
            )
            models[record.nom_modele]["total_tokens"] += record.total_tokens
            models[record.nom_modele]["total_requests"] += record.total_requests or 0

        return {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_tokens,
            "total_requests": total_requests,
            "total_success": total_success,
            "total_denials": total_denials,
            "models": models,
        }

    ################################################################
    # DELETE METHODS
    ################################################################

    def delete_by_date(self, target_date: date) -> int:
        """
        Delete all records for a specific date.

        Args:
            target_date (date): The date to delete records for.

        Returns:
            int: Number of records deleted.

        Example:
            >>> deleted = service.delete_by_date(date(2024, 1, 1))
        """
        logger.debug(f"Deleting LLM usage records for date={target_date}.")
        with self.db_manager.session() as session:
            count = (
                session.query(LLMMetrics)
                .filter(LLMMetrics.usage_date == target_date)
                .delete()
            )
            session.commit()
            logger.info(f"Deleted {count} records for {target_date}.")
            return count

    def delete_older_than(self, days: int) -> int:
        """
        Delete records older than N days.

        Args:
            days (int): Delete records older than this many days.

        Returns:
            int: Number of records deleted.

        Example:
            >>> deleted = service.delete_older_than(30)  # Keep last 30 days
        """
        cutoff_date = date.today() - timedelta(days=days)
        logger.debug(f"Deleting LLM usage records older than {cutoff_date}.")
        with self.db_manager.session() as session:
            count = (
                session.query(LLMMetrics)
                .filter(LLMMetrics.usage_date < cutoff_date)
                .delete()
            )
            session.commit()
            logger.info(f"Deleted {count} records older than {days} days.")
            return count
