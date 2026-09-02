"""CloudOptima API - FastAPI application layer.

Deployed on Render; consumed by the React dashboard on Vercel.

Endpoint groups
    /api/health, /api/meta            service + artifact status
    /api/models/*                     training metrics, prediction, explanation
    /api/providers                    multi-cloud scoreboard and arbitrage
    /api/anomaly/check                anomaly scoring
    /api/session/*                    stateful fleet simulation
    /api/ablation                     controlled component study
    /api/workload/history             the training dataset for charting
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from .catalog import (
    INSTANCE_SPECS,
    PROVIDER_SPECS,
    REGION_SPECS,
    SelectionWeights,
    score_providers,
)
from .engine import MAX_FLEET, TARGET_UTILISATION, AdvisoryEngine
from .models import CloudProvider, InstanceType, Region, Task
from .schemas import (
    AblationRequest,
    AnomalyRequest,
    InjectRequest,
    PredictRequest,
    PredictResponse,
    ProviderQuery,
    ScaleRequest,
    ScenarioRequest,
    SessionCreateRequest,
    StepRequest,
)
from .sessions import SessionStore
from .simulation import (
    STRATEGY_LABELS,
    TICK_SECONDS,
    SimulationHarness,
    split_into_tasks,
)

VERSION = "1.0.0"
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data" / "workload_history.csv"

app = FastAPI(
    title="CloudOptima API",
    description="Cloud Computing Resource Optimizer - ML + DQN + multi-cloud engine.",
    version=VERSION,
)

# The Vercel deployment origin is injected at runtime. Preview deployments get
# a generated subdomain, so the regex covers *.vercel.app rather than pinning
# one host and breaking every preview build.
_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)

store = SessionStore(artifacts_dir=ARTIFACTS)
_predictor_cache: dict[str, object] = {}
_detector_cache: dict[str, object] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _training_report() -> dict:
    path = ARTIFACTS / "training_report.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _get_predictor(algo: str):
    if algo not in _predictor_cache:
        from .ml_models import WorkloadPredictor

        try:
            _predictor_cache[algo] = WorkloadPredictor.load(ARTIFACTS, algo)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Model '{algo}' is not trained. {exc}",
            ) from exc
    return _predictor_cache[algo]


def _get_detector(method: str):
    if method not in _detector_cache:
        from .ml_models import AnomalyDetector

        try:
            _detector_cache[method] = AnomalyDetector.load(ARTIFACTS, method)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _detector_cache[method]


def _session_or_404(session_id: str):
    session = store.get(session_id)
    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired. Create a new one via POST /api/session.",
        )
    return session


def _session_payload(session) -> dict:
    alloc = session.allocator
    metrics = alloc.get_metrics()
    predicted = session.history[-1] if session.history else {}

    advisory = AdvisoryEngine(alloc).generate(
        predicted.get("predicted_cpu", metrics["cpu_used"]),
        predicted.get("predicted_ram", metrics["ram_used"]),
        anomaly=predicted.get("anomaly_detail"),
    ).as_dict()

    return {
        "session_id": session.id,
        "strategy": session.strategy,
        "strategy_label": STRATEGY_LABELS.get(session.strategy, session.strategy),
        "tick_index": session.tick_index,
        "metrics": metrics,
        "fleet": alloc.fleet_snapshot(),
        "advisory": advisory,
        "status": alloc.status(),
        "history": session.history[-120:],
        "logs": session.logs[-40:],
        "config": {
            "max_fleet": MAX_FLEET,
            "target_utilisation": TARGET_UTILISATION * 100,
            "tick_seconds": TICK_SECONDS,
        },
    }


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"service": "CloudOptima API", "version": VERSION, "docs": "/docs"}


@app.get("/api/health")
def health() -> dict:
    report = _training_report()
    required = [
        "predictor_xgboost.json", "cpu_xgboost.joblib", "ram_xgboost.joblib",
        "anomaly_isolation_forest.json",
    ]
    present = {name: (ARTIFACTS / name).exists() for name in required}
    return {
        "status": "ok" if all(present.values()) else "degraded",
        "version": VERSION,
        "artifacts_ready": all(present.values()),
        "detail": {
            "artifacts": present,
            "dataset_present": DATA.exists(),
            "trained_at": report.get("generated_at"),
            "sessions": store.stats(),
        },
    }


@app.get("/api/meta")
def meta() -> dict:
    from .ml_models import ACTION_NAMES, FEATURES

    return {
        "version": VERSION,
        "features": FEATURES,
        "rl_actions": ACTION_NAMES,
        "strategies": [{"id": k, "label": v} for k, v in STRATEGY_LABELS.items()],
        "instance_types": {
            k.value: {
                "cpu": v.cpu, "ram": v.ram,
                "base_cost_per_hour": v.base_cost_per_hour,
                "energy_efficiency": v.energy_efficiency,
                "max_power_watts": v.max_power_watts,
            } for k, v in INSTANCE_SPECS.items()
        },
        "providers": {
            k.value: {
                "base_multiplier": v.base_multiplier,
                "base_latency_ms": v.base_latency_ms,
                "volatility": v.volatility,
                "reliability": v.reliability,
                "pue": v.pue,
            } for k, v in PROVIDER_SPECS.items()
        },
        "regions": {
            k.value: {
                "cost_multiplier": v.cost_multiplier,
                "latency_offset_ms": v.latency_offset_ms,
                "carbon_intensity": v.carbon_intensity,
            } for k, v in REGION_SPECS.items()
        },
        "limits": {"max_fleet": MAX_FLEET, "tick_seconds": TICK_SECONDS},
    }


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@app.get("/api/models/metrics")
def model_metrics() -> dict:
    report = _training_report()
    if not report:
        raise HTTPException(
            status_code=503,
            detail="No training report. Run scripts/train.py to produce artifacts.",
        )
    return report


@app.post("/api/models/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    from .ml.predictor import HORIZON

    predictor = _get_predictor(req.algo)
    import numpy as np

    # A standalone prediction has no live history, so the lag features are
    # seeded from the request's own implied demand rather than a constant.
    implied = req.num_tasks * req.cpu_per_task
    implied_ram = req.num_tasks * req.ram_per_task
    row = {
        "num_tasks": req.num_tasks,
        "cpu_per_task": req.cpu_per_task,
        "ram_per_task": req.ram_per_task,
        "hour_sin": float(np.sin(2 * np.pi * req.hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * req.hour / 24.0)),
        "day_of_week": float(req.day_of_week),
        "is_weekend": 1.0 if req.day_of_week >= 5 else 0.0,
        "cpu_lag_1": implied,
        "cpu_lag_4": implied,
        "cpu_rolling_mean_4": implied,
        "cpu_rolling_std_8": 0.0,
        "ram_lag_1": implied_ram,
    }
    cpu, ram = predictor.predict(row)
    return PredictResponse(
        predicted_cpu=round(cpu, 4),
        predicted_ram=round(ram, 4),
        algo=req.algo,
        horizon_intervals=HORIZON,
        explanation=predictor.explain(row) if req.explain else None,
    )


@app.get("/api/workload/history")
def workload_history(limit: int = 288, offset: int = 0) -> dict:
    if not DATA.exists():
        raise HTTPException(
            status_code=503,
            detail="Dataset missing. Run scripts/generate_data.py.",
        )
    limit = max(1, min(limit, 2000))
    df = pd.read_csv(DATA)
    total = len(df)
    window = df.iloc[offset: offset + limit]
    return {
        "total_rows": total,
        "offset": offset,
        "limit": limit,
        "columns": list(df.columns),
        "rows": json.loads(window.to_json(orient="records")),
        "summary": {
            "cpu_mean": round(float(df["cpu_demand"].mean()), 3),
            "cpu_max": round(float(df["cpu_demand"].max()), 3),
            "ram_mean": round(float(df["ram_demand"].mean()), 3),
            "ram_max": round(float(df["ram_demand"].max()), 3),
            "burst_events": int(df["burst_onset"].sum()) if "burst_onset" in df else 0,
        },
    }


# ---------------------------------------------------------------------------
# Multi-cloud
# ---------------------------------------------------------------------------

@app.post("/api/providers/score")
def providers_score(q: ProviderQuery) -> dict:
    import time as _time

    weights = SelectionWeights(q.weight_cost, q.weight_latency, q.weight_carbon)
    rows = score_providers(
        _time.time(),
        InstanceType(q.instance_type),
        Region(q.region),
        weights,
    )
    return {
        "evaluated_at": _time.time(),
        "instance_type": q.instance_type,
        "region": q.region,
        "weights": weights.normalised().__dict__,
        "scoreboard": rows,
        "recommended": rows[0]["provider"],
        "saving_vs_worst_pct": round(
            (rows[-1]["hourly_cost"] - rows[0]["hourly_cost"])
            / max(rows[-1]["hourly_cost"], 1e-9) * 100, 2
        ),
    }


@app.get("/api/providers/price-series")
def price_series(hours: int = 24, points: int = 96) -> dict:
    """Deterministic price curve for each provider over a forward window."""
    import time as _time

    from .catalog import price_index

    points = max(8, min(points, 512))
    now = _time.time()
    step = hours * 3600 / points
    series = []
    for i in range(points):
        at = now + i * step
        series.append({
            "t": round(at, 1),
            "offset_hours": round(i * step / 3600, 3),
            **{p.value: price_index(p, at) for p in CloudProvider},
        })
    return {"hours": hours, "points": points, "series": series}


# ---------------------------------------------------------------------------
# Anomaly
# ---------------------------------------------------------------------------

@app.post("/api/anomaly/check")
def anomaly_check(req: AnomalyRequest) -> dict:
    detector = _get_detector(req.method)
    return detector.check(
        req.cpu_demand, req.ram_demand,
        cpu_prev=req.cpu_prev,
        cpu_rolling=req.cpu_rolling,
        ram_rolling=req.ram_rolling,
    )


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

@app.post("/api/session")
def create_session(req: SessionCreateRequest) -> dict:
    session = store.create(
        predictor_algo=req.predictor_algo,
        anomaly_method=req.anomaly_method,
        strategy=req.strategy,
        initial_fleet=req.initial_fleet,
        seed=req.seed,
    )
    return _session_payload(session)


@app.get("/api/session/{session_id}")
def get_session(session_id: str = PathParam(...)) -> dict:
    return _session_payload(_session_or_404(session_id))


@app.delete("/api/session/{session_id}")
def delete_session(session_id: str = PathParam(...)) -> dict:
    return {"deleted": store.delete(session_id)}


@app.post("/api/session/{session_id}/step")
def step_session(session_id: str, req: StepRequest = Body(default=StepRequest())) -> dict:
    """Advance the simulation: observe -> predict -> decide -> place -> tick."""
    session = _session_or_404(session_id)
    alloc = session.allocator

    for _ in range(req.ticks):
        # Retire the previous interval's cohort at the *start* of this one.
        # Deferring it this way leaves the fleet holding live work when the
        # response is built, so the dashboard shows the utilisation that was
        # actually served rather than an idle cluster.
        alloc.retire_completed()
        obs = session.generator.step()
        demand_cpu, demand_ram = obs["cpu_demand"], obs["ram_demand"]

        uses_prediction = session.strategy != "static_rules"
        if uses_prediction and alloc.predictor is not None:
            predicted_cpu, predicted_ram = alloc.predict_demand(
                obs["num_tasks"], obs["cpu_per_task"], obs["ram_per_task"],
                obs["hour"], obs["day_of_week"],
            )
        else:
            predicted_cpu, predicted_ram = demand_cpu, demand_ram

        anomaly = alloc.check_anomaly(demand_cpu, demand_ram) if uses_prediction else None

        # Phase 1: decide and resize. The reward for this action can only be
        # measured after the interval's work has been placed against the new
        # fleet, so scoring is deferred to phase 2 below.
        action_label = "fixed"
        reward = None
        pending_rl = None

        if session.strategy in ("rl_only", "full"):
            pending_rl = alloc.rl_begin(predicted_cpu, predicted_ram)
            action_label = pending_rl["action_name"]
        elif session.autoscaler is not None:
            result = session.autoscaler.step(
                session.tick_index, predicted_cpu, predicted_ram,
                observed_cpu=session.last_observed_cpu,
            )
            action_label = ",".join(result["actions"]) or "hold"
            if result["actions"]:
                session.log("AUTOSCALE", action_label)

        n_tasks, cpu_each, ram_each = split_into_tasks(demand_cpu, demand_ram)
        failed_before = alloc.failed_tasks
        for _ in range(n_tasks):
            alloc.allocate_task(
                Task(cpu_required=cpu_each, ram_required=ram_each,
                     duration=TICK_SECONDS),
                strategy="cost_aware" if alloc.multi_cloud else "best_fit",
            )
        failed = alloc.failed_tasks - failed_before
        if failed:
            session.log("ALERT", f"{failed} task(s) could not be placed - capacity exhausted")

        # Phase 2: score the action against the interval it actually produced.
        if pending_rl is not None:
            rl = alloc.rl_complete(
                pending_rl, predicted_cpu, predicted_ram,
                placement_failures=failed, tasks_submitted=n_tasks,
                train=req.train_rl,
            )
            reward = rl["reward"]
            session.log("DQN", f"{rl['effect']} (reward {rl['reward']:+.3f})")

        metrics = alloc.get_metrics()
        session.last_observed_cpu = metrics["cpu_used"]

        # Bill and advance the clock but keep the cohort resident.
        alloc.tick(TICK_SECONDS, retire=False)
        alloc.observe(demand_cpu, demand_ram)

        if anomaly and anomaly.get("is_anomaly"):
            session.log("ANOMALY", f"Abnormal demand pattern "
                                   f"(severity {anomaly['severity']:.2f}) via {anomaly['method']}")

        session.history.append({
            "tick": session.tick_index,
            "hour": obs["hour"],
            "demand_cpu": round(demand_cpu, 3),
            "demand_ram": round(demand_ram, 3),
            "predicted_cpu": round(predicted_cpu, 3),
            "predicted_ram": round(predicted_ram, 3),
            "capacity_cpu": metrics["cpu_capacity"],
            "utilisation": metrics["cpu_utilization"],
            "fleet_size": metrics["fleet_size"],
            "hourly_cost": metrics["hourly_cost"],
            "power_watts": metrics["power_watts"],
            "sla_compliance": metrics["sla_compliance"],
            "action": action_label,
            "reward": reward,
            "anomaly": bool(anomaly and anomaly.get("is_anomaly")),
            "anomaly_detail": anomaly,
            "tasks_failed": failed,
        })
        if len(session.history) > 600:
            del session.history[: len(session.history) - 600]
        session.tick_index += 1

    return _session_payload(session)


@app.post("/api/session/{session_id}/inject")
def inject_load(session_id: str, req: InjectRequest) -> dict:
    session = _session_or_404(session_id)
    alloc = session.allocator
    placed = failed = 0
    for _ in range(req.task_count):
        ok = alloc.allocate_task(
            Task(cpu_required=req.cpu_per_task, ram_required=req.ram_per_task,
                 duration=req.duration),
            strategy="cost_aware" if alloc.multi_cloud else "best_fit",
        )
        placed += ok
        failed += not ok
    session.log("INJECT", f"{req.task_count} tasks submitted - {placed} placed, {failed} rejected")
    payload = _session_payload(session)
    payload["injection"] = {"requested": req.task_count, "placed": placed, "failed": failed}
    return payload


@app.post("/api/session/{session_id}/scale")
def scale_fleet(session_id: str, req: ScaleRequest) -> dict:
    session = _session_or_404(session_id)
    alloc = session.allocator
    added = []
    for _ in range(req.count):
        vm = alloc.add_vm(
            InstanceType(req.instance_type),
            provider=CloudProvider(req.provider) if req.provider else None,
        )
        if vm is None:
            session.log("WARN", f"Fleet cap of {MAX_FLEET} reached - provisioning refused")
            break
        added.append(vm.id)
    if added:
        session.log("SCALE", f"Provisioned {len(added)} x {req.instance_type}: {', '.join(added)}")
    payload = _session_payload(session)
    payload["added"] = added
    return payload


@app.delete("/api/session/{session_id}/vm/{vm_id}")
def decommission(session_id: str, vm_id: str) -> dict:
    session = _session_or_404(session_id)
    ok = session.allocator.remove_vm(vm_id, reallocate=True)
    session.log("SCALE", f"Decommissioned {vm_id}" if ok
                else f"Could not decommission {vm_id} (not found or at fleet floor)")
    payload = _session_payload(session)
    payload["removed"] = ok
    return payload


@app.post("/api/session/{session_id}/fault/{vm_id}")
def simulate_fault(session_id: str, vm_id: str) -> dict:
    """Kill a node and report how many tasks survived re-placement (US-15)."""
    session = _session_or_404(session_id)
    result = session.allocator.simulate_fault(vm_id)
    if result.get("ok"):
        session.log(
            "HEAL",
            f"Node {vm_id} failed - {result['tasks_recovered']}/{result['tasks_displaced']} "
            f"tasks re-placed, {result['tasks_lost']} lost",
        )
    payload = _session_payload(session)
    payload["fault"] = result
    return payload


@app.post("/api/session/{session_id}/scenario")
def apply_scenario(session_id: str, req: ScenarioRequest) -> dict:
    """The guided demo states from the report's evolution walkthrough."""
    session = _session_or_404(session_id)
    alloc = session.allocator
    alloc.vms.clear()

    if req.scenario == "over_provisioned":
        for _ in range(8):
            alloc.add_vm(InstanceType.LARGE)
        for _ in range(4):
            alloc.allocate_task(Task(0.5, 0.5, duration=900))
        session.log("SCENARIO", "Over-provisioned: 8 large nodes serving a trivial workload")

    elif req.scenario == "under_provisioned":
        alloc.add_vm(InstanceType.SMALL)
        for _ in range(60):
            alloc.allocate_task(Task(1.0, 1.0, duration=900))
        session.log("SCENARIO", "Under-provisioned: demand surge against a single small node")

    elif req.scenario == "balanced":
        cap_target = 24.0
        while sum(v.cpu_capacity for v in alloc.vms) < cap_target:
            alloc.add_vm(InstanceType.MEDIUM)
        for _ in range(19):
            alloc.allocate_task(Task(1.0, 2.0, duration=900))
        session.log("SCENARIO", "Balanced: fleet sized to roughly the target utilisation band")

    else:  # reset
        for _ in range(3):
            alloc.add_vm(InstanceType.MEDIUM)
        session.history.clear()
        session.tick_index = 0
        session.log("SCENARIO", "Reset to a clean three-node fleet")

    return _session_payload(session)


@app.get("/api/session/{session_id}/explain")
def explain(session_id: str) -> dict:
    """SHAP attribution for the most recent prediction (US-16)."""
    session = _session_or_404(session_id)
    result = session.allocator.explain_last()
    if result.get("method") == "unavailable":
        raise HTTPException(
            status_code=409,
            detail="No prediction has been made yet in this session. Step the simulation first.",
        )
    result["prediction"] = {
        "cpu": round(session.allocator.last_prediction.get("cpu", 0.0), 4),
        "ram": round(session.allocator.last_prediction.get("ram", 0.0), 4),
    }
    return result


@app.get("/api/session/{session_id}/rl")
def rl_state(session_id: str) -> dict:
    session = _session_or_404(session_id)
    agent = session.allocator.dqn_agent
    from .ml_models import ACTION_NAMES

    return {
        "epsilon": round(agent.epsilon, 5),
        "learn_steps": agent.learn_steps,
        "env_steps": agent.env_steps,
        "replay_size": len(agent.memory),
        "loss_curve": [round(v, 6) for v in agent.loss_history[-200:]],
        "actions": ACTION_NAMES,
        "reward_curve": [
            h["reward"] for h in session.history[-200:] if h.get("reward") is not None
        ],
    }


# ---------------------------------------------------------------------------
# Ablation
# ---------------------------------------------------------------------------

@app.post("/api/ablation")
def ablation(req: AblationRequest) -> dict:
    """Run every control policy against an identical workload trace."""
    cached = ARTIFACTS / "ablation.json"
    if req.ticks == 288 and req.seed == 42 and not req.strategies and cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))

    harness = SimulationHarness(artifacts_dir=ARTIFACTS)
    try:
        return harness.ablation(ticks=req.ticks, strategies=req.strategies, seed=req.seed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/ablation/cached")
def ablation_cached() -> dict:
    cached = ARTIFACTS / "ablation.json"
    if not cached.exists():
        raise HTTPException(
            status_code=503,
            detail="No cached ablation. Run scripts/evaluate.py.",
        )
    return json.loads(cached.read_text(encoding="utf-8"))
