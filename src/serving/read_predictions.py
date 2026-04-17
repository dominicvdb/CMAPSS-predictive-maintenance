"""
Read layer for downstream consumption of prediction outputs.

Provides stable query helpers for the API and dashboard.
No training code is required to use this module.

Usage:
    from src.serving.read_predictions import get_db_engine, get_latest_run_id, get_fleet_latest, get_engine_history
"""
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


PREDICTIONS_TABLE = "fd001_val_predictions_history"

FLEET_COLUMNS = [
    "run_id",
    "engine_id",
    "cycle",
    "predicted_rul",
    "risk_bucket_predicted",
    "actual_rul",
    "abs_error",
    "prediction_timestamp",
    "model_name",
    "dataset_name",
    "split_name",
]


def get_db_engine():
    load_dotenv()
    db_user = os.getenv("POSTGRES_USER", "cmapss")
    db_password = os.getenv("POSTGRES_PASSWORD", "cmapss")
    db_name = os.getenv("POSTGRES_DB", "cmapss")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    return create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )


def get_latest_run_id(engine) -> str:
    """Return the run_id from the most recent prediction batch."""
    query = text(f"""
        SELECT run_id
        FROM {PREDICTIONS_TABLE}
        ORDER BY prediction_timestamp DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        result = conn.execute(query).scalar()
    if result is None:
        raise RuntimeError(f"No rows found in '{PREDICTIONS_TABLE}'.")
    return result


def get_fleet_latest(engine, run_id: str) -> pd.DataFrame:
    """
    Return one row per engine for the latest observed cycle in a given run.
    This is the primary view for fleet-level dashboards and API endpoints.
    """
    query = text(f"""
        SELECT {", ".join(FLEET_COLUMNS)}
        FROM {PREDICTIONS_TABLE}
        WHERE run_id = :run_id
          AND is_latest_cycle = TRUE
        ORDER BY predicted_rul ASC
    """)
    df = pd.read_sql(query, engine, params={"run_id": run_id})
    return df


def get_engine_history(engine, run_id: str, engine_id: int) -> pd.DataFrame:
    """
    Return the full prediction history for one engine across all cycles in a given run.
    Ordered by cycle ascending — useful for time-series plots.
    """
    query = text(f"""
        SELECT {", ".join(FLEET_COLUMNS)}
        FROM {PREDICTIONS_TABLE}
        WHERE run_id = :run_id
          AND engine_id = :engine_id
        ORDER BY cycle ASC
    """)
    df = pd.read_sql(query, engine, params={"run_id": run_id, "engine_id": engine_id})
    return df


if __name__ == "__main__":
    engine = get_db_engine()

    run_id = get_latest_run_id(engine)
    print(f"Latest run_id: {run_id}\n")

    fleet = get_fleet_latest(engine, run_id)
    print(f"Fleet latest — {len(fleet)} engines")
    print(fleet[["engine_id", "cycle", "predicted_rul", "risk_bucket_predicted", "abs_error"]].head(10).to_string(index=False))

    sample_engine = int(fleet["engine_id"].iloc[0])
    history = get_engine_history(engine, run_id, sample_engine)
    print(f"\nEngine {sample_engine} history — {len(history)} cycles")
    print(history[["cycle", "predicted_rul", "actual_rul", "abs_error"]].tail(5).to_string(index=False))
