"""
Baseline XGBoost model for RUL prediction.

Pipeline:
  fd001_features_train -> XGBoost regressor -> evaluate on fd001_features_val
  Logs parameters, metrics, and model artifact to MLflow.

Run from repo root:
    python src/training/train_baseline.py
"""
import os

import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sqlalchemy import create_engine
from xgboost import XGBRegressor


# Columns that are identifiers or the target — never used as model inputs.
NON_FEATURE_COLS = ["engine_id", "cycle", "max_cycle", "rul"]

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "rul-baseline"


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


def load_data(engine):
    print("Loading feature tables from PostgreSQL...")
    train_df = pd.read_sql("SELECT * FROM fd001_features_train", engine)
    val_df = pd.read_sql("SELECT * FROM fd001_features_val", engine)
    print(f"  Train: {len(train_df)} rows, {train_df['engine_id'].nunique()} engines")
    print(f"  Val:   {len(val_df)} rows, {val_df['engine_id'].nunique()} engines")
    return train_df, val_df


def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def train(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model: XGBRegressor, X: pd.DataFrame, y: pd.Series, label: str) -> dict:
    preds = model.predict(X)
    # Clip negative predictions — RUL cannot be below 0.
    preds = np.clip(preds, 0, None)
    mae = mean_absolute_error(y, preds)
    rmse = root_mean_squared_error(y, preds)
    print(f"\n{label}")
    print(f"  MAE:  {mae:.2f} cycles")
    print(f"  RMSE: {rmse:.2f} cycles")
    return {"mae": mae, "rmse": rmse, "preds": preds}


def main() -> None:
    db_engine = get_engine()
    train_df, val_df = load_data(db_engine)

    feature_cols = get_feature_cols(train_df)
    print(f"\nFeatures used: {len(feature_cols)}")

    X_train = train_df[feature_cols]
    y_train = train_df["rul"]
    X_val = val_df[feature_cols]
    y_val = val_df["rul"]

    # ── MLflow setup ─────────────────────────────────────────────────────────
    # Tell MLflow where the tracking server is (our Docker container).
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    # Group this run under an experiment name. MLflow creates it if it doesn't exist.
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    print("\nTraining XGBoost...")
    # Everything inside `with mlflow.start_run()` belongs to one run.
    with mlflow.start_run(run_name="xgboost-baseline"):

        model = train(X_train, y_train)

        train_metrics = evaluate(model, X_train, y_train, "Train set")
        val_metrics = evaluate(model, X_val, y_val, "Val set")

        gap = val_metrics["mae"] - train_metrics["mae"]
        print(f"\nTrain/val MAE gap: {gap:.2f} cycles")
        if gap > 10:
            print("  (large gap — model may be overfitting)")
        else:
            print("  (gap looks reasonable)")

        # ── Log hyperparameters ───────────────────────────────────────────────
        # These are stored so you can compare runs and know exactly what
        # settings produced a given result.
        mlflow.log_params({
            "n_estimators": model.n_estimators,
            "learning_rate": model.learning_rate,
            "max_depth": model.max_depth,
            "subsample": model.subsample,
            "colsample_bytree": model.colsample_bytree,
            "window": 5,
            "n_features": len(feature_cols),
        })

        # ── Log metrics ───────────────────────────────────────────────────────
        # Prefixed with train_ / val_ so they're easy to compare side by side.
        mlflow.log_metrics({
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "mae_gap": gap,
        })

        # ── Log the model ─────────────────────────────────────────────────────
        # Saves the trained model as an artifact so you can load and serve it
        # later without retraining.
        mlflow.xgboost.log_model(model, name="model")

        print(f"\nRun logged to MLflow experiment '{MLFLOW_EXPERIMENT}'")


if __name__ == "__main__":
    main()
