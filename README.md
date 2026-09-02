# CloudOptima — Cloud Computing Resource Optimizer

Predictive, reinforcement-learned, multi-cloud resource optimisation, delivered as
a React dashboard (Vercel) over a FastAPI engine (Render).

**Aditya Ayushman Sahoo · Sarthak Kar** — B.Tech CSE (AI & ML), SRM Institute of
Science and Technology. Project 21CSP302L.

---

## What it does

An XGBoost model forecasts next-interval CPU and RAM demand from historical
workload telemetry. A Deep Q-Network consumes that forecast plus live fleet state
and chooses a **capacity headroom setpoint**; the fleet is resized to match across
AWS, Azure and GCP, with the provider for each new node chosen by a cost/latency/
carbon score. An Isolation Forest flags abnormal demand, and every prediction
carries exact TreeSHAP attribution so an operator can see why the system acted.

```
workload → preprocessing → XGBoost forecast → DQN setpoint → multi-cloud placement
              ↑                                                      │
              └──────────────── reward ← measured interval ←─────────┘
```

## Architecture

| Layer | Component | File |
|---|---|---|
| Domain | Enums + `VMInstance` / `Task` | `backend/app/models.py` |
| Catalogue | Instance specs, provider pricing, multi-cloud scoring | `backend/app/catalog.py` |
| ML | Predictor · DQN · Q-learning · anomaly detector | `backend/app/ml_models.py` (→ `app/ml/`) |
| Processing | `ResourceAllocator` · `SmartAllocator` · `AutoScaler` · `AdvisoryEngine` | `backend/app/engine.py` |
| Evaluation | Closed-loop harness + ablation | `backend/app/simulation.py` |
| Application | FastAPI service | `backend/app/main.py` |
| Presentation | React dashboard | `frontend/src/` |

## Quick start

```bash
# Backend
cd backend
python -m venv .venv && .venv/Scripts/activate      # Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt

python scripts/generate_data.py --days 30 --interval 5   # → data/workload_history.csv
python scripts/train.py                                   # → artifacts/*
python scripts/evaluate.py                                # → artifacts/ablation.json
python scripts/horizon_study.py                           # → artifacts/horizon_study.json
pytest                                                    # 58 tests

uvicorn app.main:app --reload                             # http://localhost:8000/docs
```

```bash
# Frontend (separate terminal)
cd frontend
npm install
cp .env.example .env        # VITE_API_BASE_URL=http://localhost:8000
npm run dev                 # http://localhost:5173
```

## Deployment

**Backend — Render.** The repo root carries `render.yaml`. In Render: *New →
Blueprint → connect this repository*. The build regenerates the dataset and trains
all artifacts (~5 min) rather than committing 35 MB of `.joblib` files, so the
deployed models always match the deployed code. After the frontend is live, set
the `CORS_ORIGINS` environment variable to its URL.

**Frontend — Vercel.** *New Project → import this repository →* set **Root
Directory** to `frontend`. Framework preset Vite; build `npm run build`; output
`dist`. Add one environment variable:

```
VITE_API_BASE_URL = https://<your-render-service>.onrender.com
```

Preview deployments on `*.vercel.app` are already allowed by a CORS regex in
`app/main.py`, so only the production origin needs listing.

> **Free-tier note.** Render free instances sleep after ~15 minutes idle; the
> first request then takes 30–60 s. The dashboard's ignition screen retries and
> says so rather than hanging. Simulation sessions live in memory and are lost on
> a cold start — see `app/sessions.py` for why that is deliberate.

## Measured results

Every number below is produced by `scripts/evaluate.py`, not asserted. All seven
control policies run against an **identical** workload trace per seed; only the
policy varies. See `docs/RESULTS.md` for the full table and `artifacts/ablation.json`
for the raw output.

### Validated on a real production trace

The synthetic results below are reproduced on **Bitbrains GWA-T-12** — 300 real
VMs from a production datacentre, 5-minute sampling, 30 days
(`scripts/fetch_bitbrains.py`). Full study: [docs/RESULTS-REAL-TRACE.md](docs/RESULTS-REAL-TRACE.md).

| | Synthetic | Real trace |
|---|---|---|
| Utilisation gain vs baseline | +28.7% | **+53.0%** |
| Cost reduction vs baseline | −33.6% | **−42.2%** |
| Task failure rate | 1.37% | **0.95%** |
| Multi-cloud cost saving | −12.8% | −14.3% |

**The RL result strengthens on real data. The forecasting result does not
survive it.** On production telemetry no tree ensemble beats a persistence
baseline at any horizon, and the deficit *grows* with horizon — the exact
opposite of the synthetic trend. The synthetic conclusion was an artefact of
structure the generator put in the data. That negative result is reported rather
than buried, and it is the clearest argument for validating on real traces.

### Ablation — 7 policies × 3 seeds × 24 h, identical trace per seed

| Configuration | Utilisation | Cost $/day | Latency | Fail % |
|---|---|---|---|---|
| Static rule-based *(negative control)* | 64.3% | 12.58 | 3873 s | 10.06 |
| Threshold reactive | 61.7% | 15.39 | 1720 s | 1.07 |
| **ML prediction only** *(baseline)* | 60.3% | 15.10 | 245 s | 0.24 |
| ML + multi-cloud | 60.0% | 13.17 | 245 s | 0.20 |
| Tabular Q-learning | 80.7% | 10.36 | 416 s | 5.25 |
| DQN only | 78.4% | 11.23 | 350 s | 1.62 |
| **All components combined** | **77.6%** | **10.03** | 345 s | 1.37 |

Against the ML-only baseline the full system delivers **+28.7% utilisation** and
**−33.6% cost** — but at 1.37% task failures versus 0.24%, and 41% worse
recovery latency. It is a different point on the cost/reliability frontier, not
a strict improvement, and the README says so because the measurement says so.

### Demand forecasting (held-out test block, one-step-ahead)

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| **XGBoost** | **0.9415** | 1.707 | 2.476 | 12.49% |
| Linear Regression | 0.9325 | 1.763 | 2.661 | 12.97% |
| Random Forest | 0.9291 | 1.845 | 2.726 | 12.86% |
| *Persistence baseline* | *0.9283* | *1.899* | — | — |

**A high R² here proves very little on its own** — "next interval equals this
one" already scores 0.9283. The forecaster's real justification is how its
advantage scales with horizon (3 seeds, margin over persistence):

| Horizon | XGBoost | Random Forest | Linear Regression |
|---|---|---|---|
| 5 min | −0.0009 *(1/3 seeds)* | −0.0143 *(0/3)* | +0.0026 *(2/3)* |
| **15 min** | **+0.0108 (3/3)** | +0.0042 (2/3) | +0.0106 (2/3) |
| **30 min** | **+0.0403 (3/3)** | +0.0245 (2/3) | +0.0186 (2/3) |
| **60 min** | **+0.1490 (3/3)** | +0.1413 (3/3) | +0.0401 (2/3) |

Breakeven is 15 minutes. At one 5-minute step the series is autocorrelated
enough that the learned model adds essentially nothing; it separates decisively
once it must see past the autocorrelation, and the ensembles only pull away from
the linear baseline at long horizons.

### Anomaly detection (event-scored, matched recall)

| Detector | Events | Detected | Precision | Recall | F1 |
|---|---|---|---|---|---|
| **Isolation Forest** | 26 | 22 | **0.361** | 0.846 | **0.506** |
| Z-score (4σ) | 26 | 22 | 0.244 | 0.846 | 0.379 |

### Reinforcement learning (greedy, fixed held-out seed)

| | Reward | Utilisation | Cost/day | Failures |
|---|---|---|---|---|
| Untrained | +0.602 | 72.2% | $13.80 | 0.80% |
| **Trained (selected checkpoint)** | **+0.713** | **79.1%** | **$12.00** | 1.27% |

Training is not monotone — held-out reward peaks near episode 6 and then drifts
into a policy that drops 4.72% of tasks. The deployed agent is the
best-scoring checkpoint, not the final weights.

## Deliberate departures from the original design

Each of these was a measured decision, not a shortcut. The reasoning lives in a
comment at the relevant code site.

| Change | Why |
|---|---|
| **NumPy DQN, not PyTorch** | The CPU torch wheel is ~800 MB installed / ~250 MB resident and does not fit a 512 MB free-tier container. The algorithm is unchanged. |
| **TreeSHAP via XGBoost, not the `shap` package** | XGBoost implements exact TreeSHAP natively (`pred_contribs=True`), dropping `shap` + numba + llvmlite from the image. |
| **Random search on a validation block, not Optuna on the test set** | The original tuned hyperparameters against the same data it reported scores on, which inflates them. |
| **Chronological 70/15/15 split** | A shuffled split on a time series puts future rows in training. |
| **Target is `t+1`, not `t`** | Predicting the current interval from current-interval features is fitting, not forecasting, and is useless to an autoscaler. |
| **RL actions are headroom setpoints** | Incremental add/remove actions create an exploration trap: escaping an undersized fleet needs several consecutive correct actions, each individually penalised. Measured collapse to a 1-node cluster dropping 67% of tasks. |
| **Reward measured after placement** | Scoring the action before work is placed reads an empty fleet — zero utilisation, zero failures — so the agent learns that the cheapest fleet is always best. |
| **Memory-optimised instance type added** | Every original instance type has a RAM:CPU ratio of 2.0 while the workload's is ~2.4, stranding ~17% of CPU behind exhausted memory. |
| **Fleet hard cap (40 nodes)** | The original scale-up loop had no upper bound; a bad forecast is a billing incident. |
| **RL checkpoint selected on a held-out seed** | DQN training is not monotone here: held-out reward peaks early and then drifts into a policy dropping 4.72% of tasks. Shipping the final weights would ship the worse agent. |
| **Carbon derived from provider PUE** | An earlier version derived it from `reliability`, which is not a carbon quantity and left the carbon weight with no distinguishable effect on provider choice. |
| **Streamlit → React + FastAPI** | Vercel cannot host Streamlit (it needs a persistent WebSocket server). Splitting presentation from engine is what makes the Vercel + Render deployment possible. |

## Repository layout

```
├── backend/
│   ├── app/
│   │   ├── models.py          domain types
│   │   ├── catalog.py         instances, pricing, multi-cloud scoring
│   │   ├── workload.py        synthetic workload generator
│   │   ├── ml_models.py       ML facade  →  app/ml/{predictor,dqn,qlearning,anomaly}.py
│   │   ├── engine.py          allocator, autoscaler, advisory, RL glue
│   │   ├── simulation.py      closed-loop harness + ablation
│   │   ├── sessions.py        in-memory session store
│   │   ├── schemas.py         request/response models
│   │   └── main.py            FastAPI application
│   ├── scripts/               generate_data · train · evaluate · horizon_study
│   ├── tests/                 pytest suite
│   └── requirements.txt
├── frontend/                  React + Vite dashboard
├── docs/                      RESULTS.md · API.md · DEPLOYMENT.md · REPORT-NOTES.md
├── render.yaml                Render blueprint
└── README.md
```

## API

Interactive docs at `/docs` on the running service. Principal endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Service and artifact status |
| `GET` | `/api/meta` | Catalogue, features, RL actions |
| `GET` | `/api/models/metrics` | Full training report |
| `POST` | `/api/models/predict` | Forecast + SHAP attribution |
| `POST` | `/api/providers/score` | Multi-cloud scoreboard |
| `POST` | `/api/session` | Create a simulation session |
| `POST` | `/api/session/{id}/step` | Advance the closed loop |
| `POST` | `/api/session/{id}/fault/{vm}` | Fault injection / self-healing |
| `GET` | `/api/session/{id}/explain` | SHAP for the latest prediction |
| `POST` | `/api/ablation` | Run the controlled study |

## Licence

Academic project submitted for 21CSP302L. See `docs/REPORT-NOTES.md` for the
mapping between this implementation and the written report.
