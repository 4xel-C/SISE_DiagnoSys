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
from enum import Enum
from typing import List, Optional, Union

from sqlalchemy import text

from app.config import Database, db
from app.models import LLMMetrics
from app.rag import LLMUsage, MistralModel
from app.schemas import AggregatedMetricsSchema, LLMMetricsSchema

logger = logging.getLogger(__name__)


class AggTime(Enum):
    """Aggregation levels for metrics."""

    DAILY = "daily"
    MONTHLY = "monthly"
    YEARLY = "yearly"


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
        return sum(r.total_requests for r in records)

    def get_aggregated_data(
        self,
        agg_time: Union[str, AggTime] = AggTime.DAILY,
        models: Union[List[str], List[MistralModel]] = MistralModel.all_models(),
    ) -> List[AggregatedMetricsSchema]:
        """
        Generate aggregated metrics for the stats page.

        Args:
            agg_time: Aggregation time (daily, monthly, yearly).
                       Accepts string or AggTime enum.
            models: Optional list of models to filter by.
                    Accepts strings or MistralModel enums.

        Returns:
            List[AggregatedMetricsSchema]: Aggregated metrics per period and model.

        Raises:
            ValueError: If agg_time is invalid.

        Example:
            >>> metrics = service.get_aggregated_data(AggLevel.MONTHLY, [MistralModel.MISTRAL_SMALL])
            >>> metrics = service.get_aggregated_data("monthly", ["mistral-small-latest"])
        """

        # Type validation
        level = AggTime(agg_time) if isinstance(agg_time, str) else agg_time

        if models == ["all"]:
            model_names = MistralModel.all_models()
        else:
            model_names = [
                m.value if isinstance(m, MistralModel) else m for m in models
            ]

        logger.debug(
            f"Generating LLM usage metrics for agg_level={level.value}, models={model_names}"
        )

        # Period format based on aggregation level
        period_formats = {
            AggTime.DAILY: "%Y-%m-%d",
            AggTime.MONTHLY: "%Y-%m",
            AggTime.YEARLY: "%Y",
        }
        period_format = period_formats[level]

        # Build SQL query with optional model filter
        where_clause = ""
        params = {}

        if model_names:
            placeholders = ", ".join(f":m{i}" for i in range(len(model_names)))
            where_clause = f"WHERE nom_modele IN ({placeholders})"
            params = {f"m{i}": name for i, name in enumerate(model_names)}

        sql = f"""
            SELECT
                strftime('{period_format}', usage_date) as period,
                nom_modele,
                SUM(total_input_tokens) as total_input_tokens,
                SUM(total_completion_tokens) as total_completion_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(total_requests) as total_requests,
                SUM(total_success) as total_success,
                SUM(COALESCE(total_denials, 0)) as total_denials,
                SUM(mean_response_time_ms * total_requests) / SUM(total_requests) as mean_response_time_ms,
                SUM(COALESCE(cout_total_usd, 0)) as cout_total_usd,
                SUM(COALESCE(energy_kwh, 0)) as energy_kwh,
                SUM(COALESCE(gwp_kgCO2eq, 0)) as gwp_kgCO2eq,
                SUM(COALESCE(adpe_mgSbEq, 0)) as adpe_mgSbEq,
                SUM(COALESCE(pd_mj, 0)) as pd_mj,
                SUM(COALESCE(wcf_liters, 0)) as wcf_liters
            FROM llm_usage_journalier
            {where_clause}
            GROUP BY period, nom_modele
            ORDER BY period DESC
        """

        with self.db_manager.session() as session:
            rows = session.execute(text(sql), params).fetchall()

            logger.debug(f"Found {len(rows)} aggregated records.")

            return [
                AggregatedMetricsSchema(
                    period=row.period,
                    nom_modele=row.nom_modele,
                    total_input_tokens=row.total_input_tokens or 0,
                    total_completion_tokens=row.total_completion_tokens or 0,
                    total_tokens=row.total_tokens or 0,
                    total_requests=row.total_requests or 0,
                    total_success=row.total_success or 0,
                    total_denials=row.total_denials or 0,
                    mean_response_time_ms=row.mean_response_time_ms or 0.0,
                    cout_total_usd=row.cout_total_usd or 0.0,
                    energy_kwh=row.energy_kwh or 0.0,
                    gwp_kgCO2eq=row.gwp_kgCO2eq or 0.0,
                    adpe_mgSbEq=row.adpe_mgSbEq or 0.0,
                    pd_mj=row.pd_mj or 0.0,
                    wcf_liters=row.wcf_liters or 0.0,
                )
                for row in rows
            ]

    def get_current_period_metrics(
        self,
        agg_time: Union[str, AggTime] = AggTime.DAILY,
    ) -> Optional[AggregatedMetricsSchema]:
        """
        Get total metrics for the current period (today, this month, or this year).

        Args:
            agg_time: Aggregation time determining the current period.
                - DAILY: today
                - MONTHLY: this month
                - YEARLY: this year

        Returns:
            AggregatedMetricsSchema: Total metrics for the current period, or None if no data.

        Example:
            >>> today_metrics = service.get_current_period_metrics(AggTime.DAILY)
            >>> this_month = service.get_current_period_metrics("monthly")
        """
        time = AggTime(agg_time) if isinstance(agg_time, str) else agg_time
        today = date.today()

        period_formats = {
            AggTime.DAILY: "%Y-%m-%d",
            AggTime.MONTHLY: "%Y-%m",
            AggTime.YEARLY: "%Y",
        }
        period_format = period_formats[time]
        current_period = today.strftime(period_format)

        sql = f"""
            SELECT
                strftime('{period_format}', usage_date) as period,
                SUM(total_input_tokens) as total_input_tokens,
                SUM(total_completion_tokens) as total_completion_tokens,
                SUM(total_tokens) as total_tokens,
                SUM(total_requests) as total_requests,
                SUM(total_success) as total_success,
                SUM(COALESCE(total_denials, 0)) as total_denials,
                SUM(mean_response_time_ms * total_requests) / SUM(total_requests) as mean_response_time_ms,
                SUM(COALESCE(cout_total_usd, 0)) as cout_total_usd,
                SUM(COALESCE(energy_kwh, 0)) as energy_kwh,
                SUM(COALESCE(gwp_kgCO2eq, 0)) as gwp_kgCO2eq,
                SUM(COALESCE(adpe_mgSbEq, 0)) as adpe_mgSbEq,
                SUM(COALESCE(pd_mj, 0)) as pd_mj,
                SUM(COALESCE(wcf_liters, 0)) as wcf_liters
            FROM llm_usage_journalier
            WHERE strftime('{period_format}', usage_date) = :current_period
        """

        with self.db_manager.session() as session:
            row = session.execute(
                text(sql), {"current_period": current_period}
            ).fetchone()

            if not row or row.period is None:
                return None

            return AggregatedMetricsSchema(
                period=row.period,
                nom_modele="all",
                total_input_tokens=row.total_input_tokens or 0,
                total_completion_tokens=row.total_completion_tokens or 0,
                total_tokens=row.total_tokens or 0,
                total_requests=row.total_requests or 0,
                total_success=row.total_success or 0,
                total_denials=row.total_denials or 0,
                mean_response_time_ms=row.mean_response_time_ms or 0.0,
                cout_total_usd=row.cout_total_usd or 0.0,
                energy_kwh=row.energy_kwh or 0.0,
                gwp_kgCO2eq=row.gwp_kgCO2eq or 0.0,
                adpe_mgSbEq=row.adpe_mgSbEq or 0.0,
                pd_mj=row.pd_mj or 0.0,
                wcf_liters=row.wcf_liters or 0.0,
            )

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
            model_name (str): The model used.
            usage (LLMUsage): Usage statistics including token counts and latency.
            success (bool): Whether the request was successful. Defaults to True.

        Returns:
            LLMMetricsSchema: The updated or created record.

        Example:
            >>> record = service.record_usage(
            ...     "mistral-small",
            ...     usage=LLMUsage(input_tokens=150, output_tokens=75, latency_ms=250.5),
            ...     success=True
            ... )
        """

        logger.debug(f"Recording LLM usage for model={model_name}.")
        today = date.today()

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        response_time_ms = usage.latency_ms
        cost_usd = usage.cost_usd
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
                record.cout_total_usd = (record.cout_total_usd or 0) + (cost_usd or 0)
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
                    cout_total_usd=cost_usd,
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
