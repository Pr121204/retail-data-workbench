"""Generate synthetic edge-case datasets for the batch evaluator.

Run once during development; the outputs are committed so the evaluator cases
are reproducible without regenerating. Deterministic (seeded).
"""
import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path("data/samples/edge")
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# --- all-null column (already-clean otherwise) ------------------------------
pd.DataFrame(
    {
        "order_id": [f"O{i}" for i in range(1, 21)],
        "region": ["West"] * 20,
        "revenue": [float(i * 10) for i in range(1, 21)],
        "coupon_code": [None] * 20,  # all-null column
    }
).to_csv(OUT / "allnull.csv", index=False)

# --- latin-1 encoded file (mixed-encoding edge case) -------------------------
pd.DataFrame(
    {
        "customer_id": [f"C{i}" for i in range(1, 16)],
        "name": ["Chloé", "José", "Åse", "Müller", "Søren"] * 3,
        "region": ["West"] * 15,
    }
).to_csv(OUT / "latin1.csv", index=False, encoding="latin-1")

# --- 100k-row orders file (large-file strategy) ------------------------------
n = 100_000
pd.DataFrame(
    {
        "order_id": [f"ORD-{i:06d}" for i in range(n)],
        "product_id": rng.choice(["P1", "P2", "P3", "P4", "P5"], n),
        "region": rng.choice(["North", "South", "East", "West"], n),
        "quantity": rng.integers(1, 5, n),
        "revenue": np.round(rng.uniform(10, 500, n), 2),
        "order_date": pd.to_datetime("2024-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
    }
).to_csv(OUT / "large_orders.csv", index=False)

print("edge datasets written:", [p.name for p in OUT.iterdir()])
