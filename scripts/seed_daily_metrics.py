import random
from datetime import date, timedelta

from app.config import db
from app.models import LLMMetrics

MODELS = ["mistral-small", "mistral-medium"]
DAYS = 30


def generate_daily_metrics(requests: int = 100):
    total_input = 0
    total_output = 0
    total_tokens = 0
    total_latency = 0.0
    success = 0
    denials = 0

    energy = gwp = adpe = pd = wcf = 0.0

    for _ in range(requests):
        input_tokens = random.randint(100, 2500)
        output_tokens = random.randint(50, 1500)
        latency = random.uniform(80, 1800)

        tokens = input_tokens + output_tokens
        e = tokens * random.uniform(1e-6, 4e-6)

        total_input += input_tokens
        total_output += output_tokens
        total_tokens += tokens
        total_latency += latency

        energy += e
        gwp += e * random.uniform(0.2, 0.6)
        adpe += e * random.uniform(0.01, 0.05)
        pd += e * random.uniform(3.0, 4.5)
        wcf += e * random.uniform(0.5, 2.5)

        if random.random() > 0.03:
            success += 1
        else:
            denials += 1

    return {
        "total_input_tokens": total_input,
        "total_completion_tokens": total_output,
        "total_tokens": total_tokens,
        "mean_response_time_ms": total_latency / requests,
        "total_requests": requests,
        "total_success": success,
        "total_denials": denials,
        "energy_kwh": energy,
        "gwp_kgCO2eq": gwp,
        "adpe_mgSbEq": adpe,
        "pd_mj": pd,
        "wcf_liters": wcf,
    }


def reset_table():
    with db.session() as session:
        deleted = session.query(LLMMetrics).delete()
        session.commit()
    print(f"🧹 Deleted {deleted} rows")


def seed_month():
    today = date.today()

    with db.session() as session:
        for offset in range(DAYS):
            day = today - timedelta(days=offset)

            for model in MODELS:
                metrics = generate_daily_metrics(100)

                session.add(
                    LLMMetrics(
                        nom_modele=model,
                        usage_date=day,
                        **metrics,
                    )
                )

            print(f"✅ Seeded {day}")

        session.commit()

    print("🎉 Seeding completed")


if __name__ == "__main__":
    print("helloworld")
    reset_table()
    print("reset done : seeding starting")
    seed_month()
    print("seeding done")
