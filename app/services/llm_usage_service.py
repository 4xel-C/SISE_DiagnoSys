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

from sqlalchemy import func

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

    # helper method
    def _format_temporal_label(self, temporal_axis: str, value: str) -> str:
        if temporal_axis == "W":
            jours = {
                "0": "dimanche",
                "1": "lundi",
                "2": "mardi",
                "3": "mercredi",
                "4": "jeudi",
                "5": "vendredi",
                "6": "samedi",
            }
            return jours.get(value, "inconnu")

        if temporal_axis == "M":
            # Ici, on peut garder le chiffre du jour ou le transformer en "1er", "2", ...
            return str(int(value))  # convertit "01" -> "1", "15" -> "15"

        if temporal_axis == "Y":
            mois = {
                "01": "janvier",
                "02": "février",
                "03": "mars",
                "04": "avril",
                "05": "mai",
                "06": "juin",
                "07": "juillet",
                "08": "août",
                "09": "septembre",
                "10": "octobre",
                "11": "novembre",
                "12": "décembre",
            }
            return mois.get(value, "inconnu")

        return value  # fallback

    def get_all_group_by(
        self, data_grouped_by: dict[str, str] | None = None
    ) -> dict[tuple[str, str | None], list[dict[str, float | str]]]:
        """
        Retrieve LLM usage data grouped by specified parameters (OLAP style).

        Args:
            data_grouped_by (dict[str, str] | None, optional):
            Parameters to group data by. Defaults to None.

        Raises:
            ValueError: If temporal_axis is not provided or unsupported.
        Returns:
            dict[tuple[str, str | None], list[dict]]: Grouped usage data.
        """
        logger.debug("Fetching LLM usage grouped by %s", data_grouped_by)
        if data_grouped_by is None:
            return self.get_all()

        temporal_axis = data_grouped_by.get("temporal_axis")
        model = data_grouped_by.get("model")
        print("temporal_axis :", temporal_axis)
        print("model :", model)

        if temporal_axis not in {"W", "M", "Y"}:
            raise ValueError(f"Unsupported temporal_axis: {temporal_axis}")

        # OLAP-style strftime expressions
        if temporal_axis == "W":
            # day of the week (0=Sunday, 1=Monday,...)
            time_expr = func.strftime("%w", LLMMetrics.usage_date).label("period")
        elif temporal_axis == "M":
            # day of the month (01-31)
            time_expr = func.strftime("%d", LLMMetrics.usage_date).label("period")
        else:  # "Y"
            # month (01-12)
            time_expr = func.strftime("%m", LLMMetrics.usage_date).label("period")

        # Metrics to aggregate with average
        metrics = {
            "total_input_tokens": LLMMetrics.total_input_tokens,
            "total_completion_tokens": LLMMetrics.total_completion_tokens,
            "total_tokens": LLMMetrics.total_tokens,
            "total_requests": LLMMetrics.total_requests,
            "total_success": LLMMetrics.total_success,
            "total_denials": LLMMetrics.total_denials,
            "energy_kwh": LLMMetrics.energy_kwh,
            "gwp_kgCO2eq": LLMMetrics.gwp_kgCO2eq,
            "adpe_mgSbEq": LLMMetrics.adpe_mgSbEq,
            "pd_mj": LLMMetrics.pd_mj,
            "wcf_liters": LLMMetrics.wcf_liters,
        }

        with self.db_manager.session() as session:
            select_cols = [time_expr, LLMMetrics.nom_modele.label("nom_modele")]
            group_by_cols = [time_expr, LLMMetrics.nom_modele]

            # Aggregation: average for OLAP
            metric_exprs = [func.avg(col).label(name) for name, col in metrics.items()]

            query = (
                session.query(*select_cols, *metric_exprs)
                .group_by(*group_by_cols)
                .order_by(*group_by_cols)
            )

            # Filter by model if specified
            if model is not None:
                query = query.filter(LLMMetrics.nom_modele == model)

            rows = query.all()

        result: dict[tuple[str, str | None], list[dict]] = {(temporal_axis, model): []}

        if not rows:
            return result

        for row in rows:
            row_dict = row._asdict()
            period = row_dict.pop("period")
            row_dict["period"] = self._format_temporal_label(temporal_axis, period)

            # Keep model_name
            row_dict["model_name"] = (
                row_dict.pop("nom_modele") if "nom_modele" in row_dict else model
            )

            result[(temporal_axis, model)].append(row_dict)

        logger.debug(
            "Grouped LLM usage result: %s",
            result[(temporal_axis, model)][0]
            if result[(temporal_axis, model)]
            else "No data",
        )
        return result

    def get_aggregated_kpi(
        self, data_grouped_by: dict[str, str] | None = None
    ) -> dict[tuple[str, str | None], dict]:
        logger.debug("Fetching aggregated LLM KPI grouped by %s", data_grouped_by)

        if data_grouped_by is None:
            return self.get_all()

        temporal_axis = data_grouped_by.get("temporal_axis")
        model_grouping = data_grouped_by.get("model")

        if temporal_axis is None:
            raise ValueError("temporal_axis must be provided in data_grouped_by")

        temporal_mapping = {
            "W": "%Y-%W",
            "M": "%Y-%m",
            "Y": "%Y",
        }

        if temporal_axis not in temporal_mapping:
            raise ValueError(f"Unsupported temporal_axis: {temporal_axis}")

        period_expr = func.strftime(
            temporal_mapping[temporal_axis], LLMMetrics.usage_date
        ).label("period")

        # Tous les KPI à sommer par période
        kpis = {
            "gwp_kgCO2eq": LLMMetrics.gwp_kgCO2eq,
            "wcf_liters": LLMMetrics.wcf_liters,
            "adpe_mgSbEq": LLMMetrics.adpe_mgSbEq,
            "energy_kwh": LLMMetrics.energy_kwh,
            "total_requests": LLMMetrics.total_requests,
        }

        query_columns = [period_expr]

        for name, column in kpis.items():
            query_columns.append(func.sum(column).label(name))

        if model_grouping:
            query_columns.insert(0, LLMMetrics.nom_modele)

        with self.db_manager.session() as session:
            query = session.query(*query_columns).group_by(period_expr)

            if model_grouping:
                query = query.group_by(LLMMetrics.nom_modele)

            results = query.all()

        if not results:
            return {}

        # 🔹 Post-agrégation : moyenne par période
        nb_periods = len(results)

        summed = {kpi: 0.0 for kpi in kpis}

        for row in results:
            for kpi in summed:
                summed[kpi] += getattr(row, kpi)

        averaged = {kpi: summed[kpi] / nb_periods for kpi in summed}
        print("averaged kpi :", averaged)
        return {(temporal_axis, model_grouping): averaged}

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
