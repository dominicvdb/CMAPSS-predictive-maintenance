"""
Feature engineering script for FD001.

Reads fd001_train_labeled from PostgreSQL and produces:
  - fd001_features_train  (80 engines)
  - fd001_features_val    (20 engines)

Each table contains:
  engine_id, cycle, max_cycle, rul,
  op_setting_1-3, all sensor_*,
  rolling mean / std / delta over last 5 cycles for selected sensors.

Run from repo root:
    python src/features/build_features.py
"""
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ── Sensors chosen after running scripts/dev/inspect_sensors.py ─────────────
# Update this list based on what inspect_sensors.py recommends for your data.
ROLLING_SENSORS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_11",
    "sensor_12",
    "sensor_15",
    "sensor_20",
    "sensor_21",
]

WINDOW = 5  # cycles to look back for rolling features

TRAIN_FRAC = 0.8
RANDOM_SEED = 42


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


def add_rul(df: pd.DataFrame) -> pd.DataFrame:
    max_cycle = df.groupby("engine_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycle, on="engine_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    return df


def add_rolling_features(df: pd.DataFrame, sensors: list, window: int) -> pd.DataFrame:
    # Sort so rolling windows are in the right order.
    df = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)

    for sensor in sensors:
        grouped = df.groupby("engine_id")[sensor]

        # Rolling mean: average value over the last `window` cycles.
        df[f"{sensor}_rmean{window}"] = grouped.transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )

        # Rolling std: how much the sensor fluctuates over the last `window` cycles.
        # min_periods=2 because std is undefined for a single value.
        df[f"{sensor}_rstd{window}"] = grouped.transform(
            lambda s: s.rolling(window, min_periods=2).std()
        )

        # Delta: difference between current value and value `window` cycles ago.
        # Positive = sensor is rising, negative = falling.
        df[f"{sensor}_delta{window}"] = grouped.transform(
            lambda s: s.diff(window)
        )

    return df


def split_on_engines(df: pd.DataFrame, train_frac: float, seed: int):
    engine_ids = df["engine_id"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(engine_ids)

    n_train = int(len(engine_ids) * train_frac)
    train_engines = engine_ids[:n_train]
    val_engines = engine_ids[n_train:]

    train_df = df[df["engine_id"].isin(train_engines)].reset_index(drop=True)
    val_df = df[df["engine_id"].isin(val_engines)].reset_index(drop=True)

    return train_df, val_df


def write_table(df: pd.DataFrame, table_name: str, engine) -> None:
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
    print(f"  Wrote {count} rows to '{table_name}'")


def main() -> None:
    engine = get_engine()

    print("Loading fd001_train_labeled...")
    df = pd.read_sql("SELECT * FROM fd001_train_labeled", engine)
    print(f"  {len(df)} rows, {df['engine_id'].nunique()} engines")

    print("\nAdding rolling features...")
    df = add_rolling_features(df, ROLLING_SENSORS, WINDOW)
    print(f"  Columns now: {len(df.columns)}")

    print("\nSplitting on engines (80/20)...")
    train_df, val_df = split_on_engines(df, TRAIN_FRAC, RANDOM_SEED)
    print(f"  Train: {train_df['engine_id'].nunique()} engines, {len(train_df)} rows")
    print(f"  Val:   {val_df['engine_id'].nunique()} engines, {len(val_df)} rows")

    print("\nWriting to PostgreSQL...")
    write_table(train_df, "fd001_features_train", engine)
    write_table(val_df, "fd001_features_val", engine)

    print("\nSample — engine 1, last 5 rows:")
    sample = train_df[train_df["engine_id"] == 1][
        ["engine_id", "cycle", "rul"] + [c for c in train_df.columns if "rmean" in c or "delta" in c]
    ].tail()
    print(sample.to_string())


if __name__ == "__main__":
    main()
