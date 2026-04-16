"""
Sensor inspection script.

Loads fd001_train_labeled from PostgreSQL and prints:
  1. Standard deviation of each sensor  — to spot near-constant (dead) sensors
  2. Pearson correlation of each sensor with RUL  — to spot informative sensors

Run from repo root:
    python scripts/dev/inspect_sensors.py
"""
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine


def get_engine():
    load_dotenv()
    db_user = os.getenv("POSTGRES_USER", "cmapss")
    db_password = os.getenv("POSTGRES_PASSWORD", "cmapss")
    db_name = os.getenv("POSTGRES_DB", "cmapss")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    return create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )


def main():
    engine = get_engine()

    print("Loading fd001_train_labeled...")
    df = pd.read_sql("SELECT * FROM fd001_train_labeled", engine)
    print(f"  {len(df)} rows, {df['engine_id'].nunique()} engines\n")

    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]

    # ── 1. Standard deviation ────────────────────────────────────────────────
    std = df[sensor_cols].std().sort_values()
    print("=" * 50)
    print("Sensor standard deviations (ascending)")
    print("Low std = near-constant = likely useless")
    print("=" * 50)
    print(std.to_string())
    print()

    # ── 2. Correlation with RUL ──────────────────────────────────────────────
    corr = df[sensor_cols].corrwith(df["rul"]).rename("corr_with_rul")
    corr_sorted = corr.abs().sort_values(ascending=False)
    print("=" * 50)
    print("Absolute correlation with RUL (descending)")
    print("Higher = more predictive of remaining life")
    print("=" * 50)
    print(corr_sorted.to_string())
    print()

    # ── 3. Summary recommendation ────────────────────────────────────────────
    LOW_STD_THRESHOLD = 0.1
    LOW_CORR_THRESHOLD = 0.5

    dead = std[std < LOW_STD_THRESHOLD].index.tolist()
    weak = corr_sorted[corr_sorted < LOW_CORR_THRESHOLD].index.tolist()
    drop = sorted(set(dead) | set(weak))
    keep = sorted(set(sensor_cols) - set(drop))

    print("=" * 50)
    print("Recommendation")
    print("=" * 50)
    print(f"Near-constant (std < {LOW_STD_THRESHOLD}):  {dead}")
    print(f"Low correlation with RUL (|r| < {LOW_CORR_THRESHOLD}):  {weak}")
    print(f"\nSuggested sensors to drop:  {drop}")
    print(f"Suggested sensors to keep for rolling features:  {keep}")


if __name__ == "__main__":
    main()
