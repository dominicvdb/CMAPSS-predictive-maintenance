from pathlib import Path
import os

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv


def main() -> None:
    load_dotenv()

    # Important:
    # This script runs on your HOST machine, not inside Docker.
    # So the database host is localhost, not "postgres".
    db_user = os.getenv("POSTGRES_USER", "cmapss")
    db_password = os.getenv("POSTGRES_PASSWORD", "cmapss")
    db_name = os.getenv("POSTGRES_DB", "cmapss")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")

    engine = create_engine(
        f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    )

    data_path = Path("data/raw/train_FD001.txt")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset file at: {data_path}")

    # C-MAPSS files are whitespace-separated with no header.
    df = pd.read_csv(data_path, sep=r"\s+", header=None, engine="python")

    # Some versions include extra empty columns at the end.
    # FD001 should have 26 useful columns:
    # 2 id columns + 3 operational settings + 21 sensor columns
    df = df.iloc[:, :26]

    columns = [
        "engine_id",
        "cycle",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
    ] + [f"sensor_{i}" for i in range(1, 22)]

    df.columns = columns

    print(f"Loaded dataframe with shape: {df.shape}")
    print(df.head())

    # For now: replace the table each time so iteration is easy.
    table_name = "raw_fd001_train"
    df.to_sql(table_name, engine, if_exists="replace", index=False)

    with engine.connect() as conn:
        row_count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
        engine_count = conn.execute(
            text(f"SELECT COUNT(DISTINCT engine_id) FROM {table_name}")
        ).scalar()

    print(f"Inserted {row_count} rows into table '{table_name}'")
    print(f"Number of distinct engines: {engine_count}")


if __name__ == "__main__":
    main()