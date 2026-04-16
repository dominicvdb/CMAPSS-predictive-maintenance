# Predictive Maintenance on NASA C-MAPSS

An end-to-end predictive maintenance platform built on the NASA C-MAPSS turbofan engine dataset. The goal is to predict **Remaining Useful Life (RUL)** for aircraft engines and surface actionable maintenance priorities through a production-style ML stack.

This is a portfolio project designed to demonstrate the full lifecycle: data ingestion → feature engineering → model training → API serving → dashboard → CI/CD → monitoring.

---

## Problem statement

Unplanned engine failures are costly. Given sensor readings from turbofan engines over time, can we predict how many cycles remain before failure — and flag engines that need maintenance soon?

**Modeling objective:** RUL regression on the FD001 subset of C-MAPSS (single operating condition, one fault mode).

**Business translation:**

| Risk category | Predicted RUL |
|---|---|
| High risk | ≤ 30 cycles |
| Medium risk | 31–60 cycles |
| Low risk | > 60 cycles |

---

## Architecture

```
Raw C-MAPSS files
    → ingestion scripts
    → PostgreSQL
    → feature pipeline
    → training pipeline (scikit-learn / XGBoost)
    → MLflow tracking + model artifacts
    → FastAPI inference service
    → prediction table in PostgreSQL
    → Streamlit dashboard
    → Evidently monitoring reports
```

---

## Stack

| Layer | Technology |
|---|---|
| Data store | PostgreSQL 16 |
| Experiment tracking | MLflow |
| ML | scikit-learn / XGBoost / LightGBM |
| API | FastAPI |
| Dashboard | Streamlit |
| Infrastructure | Docker Compose |
| CI/CD | GitHub Actions |
| Monitoring | Evidently |
| Code quality | Ruff, Black, pytest |

---

## Current status

- [x] Docker Compose stack — PostgreSQL + MLflow running locally
- [x] Raw data ingested — FD001 training trajectories loaded into PostgreSQL (`raw_fd001_train`)
- [ ] Feature pipeline (rolling stats, delta features, normalized sensors)
- [ ] Baseline model + MLflow experiment logging
- [ ] FastAPI inference endpoints
- [ ] Streamlit fleet dashboard
- [ ] GitHub Actions CI
- [ ] Evidently monitoring report

---

## Quickstart

### Prerequisites

- Docker Desktop
- Python 3.11+
- conda or venv

### 1. Clone and configure

```bash
git clone https://github.com/dominicvdb/CMAPSS-predictive-maintenance.git
cd CMAPSS-predictive-maintenance
cp env.example .env
# edit .env with your preferred credentials
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

### 4. Ingest data

```bash
python src/ingestion/ingest_fd001.py
```

This loads the FD001 training trajectories into the `raw_fd001_train` table in PostgreSQL.

---

## Dataset

NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation) — four subsets with varying operating conditions and fault modes. This project starts with **FD001**: single operating condition, single fault mode, 100 training engines.

Each row is one engine cycle with:
- 2 identifier columns (engine ID, cycle number)
- 3 operational setting columns
- 21 sensor readings

---

## Repository structure

```
├── api/                  # FastAPI app (coming Week 2)
├── dashboard/            # Streamlit app (coming Week 2)
├── monitoring/           # Evidently reports (coming Week 3)
├── src/
│   ├── ingestion/        # Data loading scripts
│   ├── features/         # Feature engineering (coming Week 1)
│   ├── training/         # Model training (coming Week 1)
│   └── utils/
├── tests/                # pytest suite (coming Week 3)
├── scripts/dev/          # Local dev utilities
├── mlflow/               # MLflow Docker config
├── docker-compose.yml
├── requirements.txt
└── env.example
```

---

## Roadmap

**Week 1** — Data foundation + baseline model
**Week 2** — FastAPI + Streamlit dashboard MVP
**Week 3** — CI/CD, tests, and monitoring
**Week 4** — Business polish and portfolio packaging
