"""
LLM Usage Seed Script.

Generates dummy LLM usage data for testing purposes.
Run with: python -m scripts.seed_llm_usage
"""

import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import db
from app.models import LLMMetrics

# Models to simulate
MODELS = [
    "mistral-small-latest",
    "mistral-large-latest",
    "ministral-8b-latest",
]


def generate_random_usage(target_date: date, model_name: str) -> dict:
    """Generate random usage data for a given date and model."""
    # Random number of requests for the day (5-50)
    total_requests = random.randint(5, 50)
    total_success = random.randint(int(total_requests * 0.9), total_requests)
    total_denials = total_requests - total_success

    # Token counts (vary by model size)
    if "large" in model_name:
        input_mult = 1.5
        output_mult = 2.0
    elif "small" in model_name:
        input_mult = 1.0
        output_mult = 1.0
    else:
        input_mult = 0.8
        output_mult = 0.7

    total_input_tokens = int(random.randint(500, 2000) * total_requests * input_mult)
    total_completion_tokens = int(
        random.randint(200, 800) * total_requests * output_mult
    )
    total_tokens = total_input_tokens + total_completion_tokens

    # Response time (ms) - varies by model
    base_response_time = {"large": 1500, "small": 500, "8b": 300}
    for key, base in base_response_time.items():
        if key in model_name:
            mean_response_time = base + random.uniform(-100, 200)
            break
    else:
        mean_response_time = 800 + random.uniform(-100, 200)

    # Cost metrics (based on model pricing per 1M tokens)
    cost_per_1m = {
        "mistral-small-latest": (0.2, 0.6),
        "mistral-large-latest": (2.0, 6.0),
        "ministral-8b-latest": (0.1, 0.1),
    }
    input_cost, output_cost = cost_per_1m.get(model_name, (0.2, 0.6))
    cout_total_usd = round(
        (total_input_tokens / 1_000_000 * input_cost)
        + (total_completion_tokens / 1_000_000 * output_cost),
        6,
    )

    # Environmental metrics (scaled by tokens)
    token_factor = total_tokens / 10000
    energy_kwh = round(0.001 * token_factor * random.uniform(0.8, 1.2), 6)
    gwp_kgCO2eq = round(0.0005 * token_factor * random.uniform(0.8, 1.2), 6)
    adpe_mgSbEq = round(0.00001 * token_factor * random.uniform(0.8, 1.2), 8)
    pd_mj = round(0.01 * token_factor * random.uniform(0.8, 1.2), 6)
    wcf_liters = round(0.05 * token_factor * random.uniform(0.8, 1.2), 6)

    return {
        "nom_modele": model_name,
        "total_input_tokens": total_input_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "mean_response_time_ms": round(mean_response_time, 2),
        "total_requests": total_requests,
        "total_success": total_success,
        "total_denials": total_denials,
        "cout_total_usd": cout_total_usd,
        "energy_kwh": energy_kwh,
        "gwp_kgCO2eq": gwp_kgCO2eq,
        "adpe_mgSbEq": adpe_mgSbEq,
        "pd_mj": pd_mj,
        "wcf_liters": wcf_liters,
        "usage_date": target_date,
    }


def seed_llm_usage(days: int = 30) -> None:
    """
    Insert dummy LLM usage records for the past N days.

    Args:
        days: Number of days to generate data for (default: 30).
    """
    today = date.today()
    records_created = 0

    with db.session() as session:
        for day_offset in range(days):
            target_date = today - timedelta(days=day_offset)

            # Each model has 70% chance of being used on any given day
            for model_name in MODELS:
                if random.random() < 0.7:
                    # Check if record already exists
                    existing = (
                        session.query(LLMMetrics)
                        .filter(
                            LLMMetrics.usage_date == target_date,
                            LLMMetrics.nom_modele == model_name,
                        )
                        .first()
                    )

                    if not existing:
                        data = generate_random_usage(target_date, model_name)
                        record = LLMMetrics(**data)
                        session.add(record)
                        records_created += 1

        session.commit()

    print(f"Created {records_created} LLM usage records over {days} days.")


def main():
    """Generate dummy LLM usage data."""
    print("Seeding LLM usage data...")
    seed_llm_usage(days=60)
    print("Done!")


if __name__ == "__main__":
    main()
