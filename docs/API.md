# CloudOptima API

Base URL: the Render service root. Interactive OpenAPI docs at `/docs`.

All responses are JSON. Errors use the standard FastAPI shape and carry an
actionable `detail`:

```json
{ "detail": "Model 'xgboost' is not trained. Run scripts/train.py." }
```

| Status | Meaning |
|---|---|
| `422` | Request failed schema validation |
| `404` | Session not found or expired (create a new one) |
| `409` | Operation not valid in the current state (e.g. explain before any prediction) |
| `503` | Model artifacts or dataset missing on the server |

---

## Service

### `GET /api/health`
Service status and which artifacts are present. Use this to wake a sleeping
free-tier instance.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "artifacts_ready": true,
  "detail": {
    "artifacts": { "predictor_xgboost.json": true, "cpu_xgboost.joblib": true },
    "dataset_present": true,
    "trained_at": "2026-09-02T15:38:11",
    "sessions": { "active_sessions": 2, "max_sessions": 200, "ttl_seconds": 2700 }
  }
}
```

### `GET /api/meta`
Instance catalogue, provider parameters, region carbon intensities, the predictor
feature list, the RL action names and the available strategies.

---

## Models

### `GET /api/models/metrics`
The full `training_report.json`: per-algorithm validation and test metrics for
both targets, the persistence baseline, anomaly detector scores, the DQN reward
curve and the Q-learning summary.

### `POST /api/models/predict`

```json
{
  "num_tasks": 40, "cpu_per_task": 0.42, "ram_per_task": 0.9,
  "hour": 14, "day_of_week": 2, "algo": "xgboost", "explain": true
}
```

```json
{
  "predicted_cpu": 17.42, "predicted_ram": 38.9,
  "algo": "xgboost", "horizon_intervals": 1,
  "explanation": {
    "method": "treeshap-exact",
    "base_value": 14.68,
    "contributions": [
      { "feature": "num_tasks", "value": 40.0, "contribution": 2.91 },
      { "feature": "cpu_lag_1", "value": 16.8, "contribution": -0.44 }
    ]
  }
}
```

`contributions` are exact SHAP values: `base_value + Σ contributions == predicted_cpu`.
For `algo=lr` the method is `linear-shap-exact`; for `algo=rf` it degrades to
`impurity-importance-approx` and is labelled as such — sklearn has no exact
TreeSHAP and the response says so rather than implying it does.

A standalone prediction has no session history, so lag features are seeded from
the request's own implied demand. For attribution against live state use
`GET /api/session/{id}/explain`.

### `GET /api/workload/history?limit=288&offset=0`
Rows from the training dataset, for charting.

### `GET /api/workload/forecastability?limit=0`

Answers "is a learned forecaster worth building for this workload?" — **without
training anything**. `limit` restricts to the most recent N intervals; 0 uses the
whole dataset.

```json
{
  "verdict": "persistence_sufficient",
  "diff_acf1": 0.1728,
  "level_acf1": 0.984,
  "cv": 0.5727,
  "reason": "Demand behaves like a random walk ..."
}
```

`verdict` is `model_likely_helps`, `persistence_sufficient` or `inconclusive`.
The decision rests on **`diff_acf1`**, the lag-1 autocorrelation of the first
difference — near zero means a random walk, where "next equals current" is
already optimal and no model can beat it; strongly negative means changes
reverse, which persistence structurally cannot exploit.

Deliberately **not** `level_acf1`: that is above 0.84 on every workload measured,
including the random walk, which is why a naive baseline scores R² > 0.9 here and
why R² alone says very little. See
[RESULTS-CROSS-DATASET.md](RESULTS-CROSS-DATASET.md) for the calibration.

---

## Multi-cloud

### `POST /api/providers/score`

```json
{ "instance_type": "medium", "region": "US-East",
  "weight_cost": 0.55, "weight_latency": 0.25, "weight_carbon": 0.20 }
```

Returns a ranked scoreboard. Cost, latency and carbon are min-max normalised
across the candidate set before combining; **lower score wins**.

### `GET /api/providers/price-series?hours=24&points=96`
Forward price curve per provider. Prices are deterministic for a given
(provider, 5-minute bucket), so repeated calls agree — the arbitrage
recommendation does not reshuffle on refresh.

---

## Anomaly

### `POST /api/anomaly/check`

```json
{ "cpu_demand": 48.2, "ram_demand": 96.0,
  "cpu_prev": 15.1, "cpu_rolling": 14.8, "ram_rolling": 35.2,
  "method": "isolation_forest" }
```

Context (`cpu_prev`, `cpu_rolling`, `ram_rolling`) is what makes a burst
separable from a normal afternoon peak. Omitting it degrades detection to level
thresholding.

---

## Sessions

Each session owns an isolated simulated fleet. Sessions are in-memory with a
45-minute idle TTL and are lost on a service restart.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/session` | Create (`strategy`, `predictor_algo`, `anomaly_method`, `initial_fleet`, `seed`) |
| `GET` | `/api/session/{id}` | Current state |
| `DELETE` | `/api/session/{id}` | Destroy |
| `POST` | `/api/session/{id}/step` | Advance `ticks` intervals of the closed loop |
| `POST` | `/api/session/{id}/inject` | Submit ad-hoc tasks |
| `POST` | `/api/session/{id}/scale` | Provision nodes manually |
| `DELETE` | `/api/session/{id}/vm/{vm_id}` | Decommission (tasks are re-placed) |
| `POST` | `/api/session/{id}/fault/{vm_id}` | Kill a node; reports recovered vs lost tasks |
| `POST` | `/api/session/{id}/scenario` | `over_provisioned` · `under_provisioned` · `balanced` · `reset` |
| `GET` | `/api/session/{id}/explain` | SHAP for the latest prediction |
| `GET` | `/api/session/{id}/rl` | Agent epsilon, replay size, loss and reward curves |

Every session-mutating endpoint returns the **full session payload**, so the
dashboard never needs a follow-up read:

```json
{
  "session_id": "…", "strategy": "full", "tick_index": 12,
  "metrics": { "cpu_utilization": 74.2, "hourly_cost": 0.41, "sla_compliance": 100.0, "…": "…" },
  "fleet": [ { "id": "vm-0007", "provider": "GCP", "type": "medium", "cpu_utilization": 81.2 } ],
  "advisory": { "warnings": [], "recommendations": [], "potential_hourly_saving": 0.0 },
  "history": [ { "tick": 11, "demand_cpu": 17.9, "predicted_cpu": 18.4, "action": "headroom_1.30x" } ],
  "logs": [ { "t": 3600.0, "source": "DQN", "message": "headroom 1.30x -> 24 cores (+1/-0, 6 nodes)" } ]
}
```

### Strategies

| id | Control policy |
|---|---|
| `static_rules` | Fixed fleet, never resizes (negative control) |
| `threshold_reactive` | Scales after measured utilisation crosses a threshold, with a provisioning delay |
| `ml_predictive` | Scales on the ML forecast |
| `multicloud_only` | `ml_predictive` + provider selection |
| `q_learning` | Tabular Q-learning over discretised state |
| `rl_only` | DQN, single provider |
| `full` | DQN + multi-cloud + anomaly detection |

---

## Ablation

### `POST /api/ablation`
```json
{ "ticks": 288, "seed": 42, "strategies": null }
```
Runs every policy on an identical trace. The default parameters are served from
`artifacts/ablation.json` when present; anything else runs live and can take a
minute or two.

### `GET /api/ablation/cached`
The pre-computed study written by `scripts/evaluate.py`. Returns `503` if the
evaluation has not been run.
