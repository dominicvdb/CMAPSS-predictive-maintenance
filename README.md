# Predictive Maintenance on NASA C-MAPSS

An end-to-end predictive maintenance pipeline built on the NASA C-MAPSS turbofan engine dataset. Predicts **Remaining Useful Life (RUL)** and surfaces maintenance risk categories through a production-style ML stack.

---

## Problem statement

Given sensor readings from turbofan engines over time, predict how many cycles remain before failure and flag engines that need maintenance soon.

**Risk categories:**

| Category | Predicted RUL |
|---|---|
| High risk | ≤ 30 cycles |
| Medium risk | 31–60 cycles |
| Low risk | > 60 cycles |

---

## Stack

| Layer | Technology |
|---|---|
| Data store | PostgreSQL 16 |
| Experiment tracking | MLflow |
| ML | XGBoost |
| Infrastructure | Docker Compose |
| Language | Python 3.11 |

---

## Current status

- [x] Docker Compose stack — PostgreSQL + MLflow
- [x] Raw FD001 data ingested into PostgreSQL (`raw_fd001_train`)
- [x] RUL labels computed — `fd001_train_labeled`
- [x] Feature engineering — rolling mean, std, delta over 5 cycles for 9 sensors; 80/20 engine-level train/val split (`fd001_features_train`, `fd001_features_val`)
- [x] Baseline XGBoost model — val MAE ~30 cycles, val RMSE ~43 cycles
- [x] MLflow experiment logging — params, metrics, model artifact, feature list
- [x] Validation predictions written to PostgreSQL (`fd001_val_predictions_history`) with risk buckets, `is_latest_cycle` flag, and MLflow `run_id`
- [ ] FastAPI inference service
- [ ] Streamlit fleet dashboard
- [ ] GitHub Actions CI
- [ ] Evidently monitoring report

---

## Quickstart

### Prerequisites

- Docker Desktop
- Python 3.11+ (conda recommended)

### 1. Clone and configure

```bash
git clone https://github.com/dominicvdb/CMAPSS-predictive-maintenance.git
cd CMAPSS-predictive-maintenance
cp env.example .env
```

### 2. Start infrastructure

```bash
docker compose up -d
```

- PostgreSQL → `localhost:5432`
- MLflow UI → `http://localhost:5000`

### 3. Set up Python environment

```bash
conda create -n cmapss python=3.11
conda activate cmapss
pip install -r requirements.txt
```

### 4. Run the pipeline

```bash
python src/ingestion/ingest_fd001.py       # load raw data
python src/features/build_features.py     # compute RUL labels + features
python src/training/train_baseline.py     # train, evaluate, log to MLflow
```

---

## Pipeline

```
Raw C-MAPSS files
    → PostgreSQL (raw_fd001_train)
    → RUL labels + feature engineering (fd001_features_train/val)
    → XGBoost training
    → MLflow (params, metrics, model artifact, feature list)
    → Validation predictions (fd001_val_predictions_history)
```

---

## Repository structure

```
├── src/
│   ├── ingestion/        # Raw data → PostgreSQL
│   ├── features/         # Feature engineering + train/val split
│   └── training/         # Model training + MLflow logging
├── scripts/dev/          # Local inspection utilities
├── api/                  # FastAPI (coming)
├── dashboard/            # Streamlit (coming)
├── monitoring/           # Evidently (coming)
├── tests/                # pytest (coming)
├── mlflow/               # MLflow Docker config
├── docker-compose.yml
├── requirements.txt
└── env.example
```
