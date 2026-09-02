"""API contract tests.

Tests that need trained artifacts skip rather than fail when the artifacts are
absent, so the suite is meaningful on a clean checkout before `train.py` runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.main import app  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
TRAINED = (ARTIFACTS / "predictor_xgboost.json").exists()
needs_models = pytest.mark.skipif(not TRAINED, reason="run scripts/train.py first")

client = TestClient(app)


def test_health_reports_artifact_state():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "artifacts" in body["detail"]


def test_meta_lists_catalogue_and_actions():
    body = client.get("/api/meta").json()
    assert len(body["features"]) >= 8
    assert len(body["rl_actions"]) >= 3
    assert {"AWS", "Azure", "GCP"} <= set(body["providers"])
    assert "memory" in body["instance_types"]


def test_provider_scoring_is_ranked():
    r = client.post("/api/providers/score", json={"instance_type": "medium", "region": "US-East"})
    assert r.status_code == 200
    board = r.json()
    assert board["recommended"] in ("AWS", "Azure", "GCP")
    assert [row["rank"] for row in board["scoreboard"]] == [1, 2, 3]


def test_price_series_is_stable_between_calls():
    """The series is anchored to 'now', so two calls a moment apart are not
    bitwise identical - the property that matters is that the curve does not
    visibly jump when a user reloads the page. Exact reproducibility for a
    fixed timestamp is covered in test_engine.
    """
    a = client.get("/api/providers/price-series?hours=6&points=12").json()
    b = client.get("/api/providers/price-series?hours=6&points=12").json()
    for pa, pb in zip(a["series"], b["series"]):
        for provider in ("AWS", "Azure", "GCP"):
            assert pa[provider] == pytest.approx(pb[provider], rel=1e-3)


def test_unknown_session_returns_404_with_guidance():
    r = client.get("/api/session/does-not-exist")
    assert r.status_code == 404
    assert "POST /api/session" in r.json()["detail"]


def test_prediction_validates_its_input():
    r = client.post("/api/models/predict", json={"num_tasks": 10, "hour": 99, "day_of_week": 0})
    assert r.status_code == 422


@needs_models
def test_prediction_returns_a_forecast_and_explanation():
    r = client.post("/api/models/predict", json={
        "num_tasks": 40, "cpu_per_task": 0.42, "ram_per_task": 0.9,
        "hour": 14, "day_of_week": 2, "algo": "xgboost", "explain": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_cpu"] > 0
    assert body["explanation"]["method"] == "treeshap-exact"
    assert len(body["explanation"]["contributions"]) >= 8


@needs_models
def test_session_lifecycle_and_stepping():
    created = client.post("/api/session", json={"strategy": "full", "initial_fleet": 3}).json()
    sid = created["session_id"]
    assert created["metrics"]["fleet_size"] == 3

    stepped = client.post(f"/api/session/{sid}/step", json={"ticks": 3}).json()
    assert stepped["tick_index"] == 3
    assert len(stepped["history"]) == 3
    assert stepped["history"][-1]["predicted_cpu"] > 0

    explained = client.get(f"/api/session/{sid}/explain")
    assert explained.status_code == 200
    assert explained.json()["contributions"]

    rl = client.get(f"/api/session/{sid}/rl").json()
    assert rl["replay_size"] > 0

    assert client.delete(f"/api/session/{sid}").json()["deleted"] is True


@needs_models
def test_scenarios_move_the_fleet():
    sid = client.post("/api/session", json={"strategy": "full"}).json()["session_id"]

    over = client.post(f"/api/session/{sid}/scenario",
                       json={"scenario": "over_provisioned"}).json()
    under = client.post(f"/api/session/{sid}/scenario",
                        json={"scenario": "under_provisioned"}).json()

    assert over["metrics"]["fleet_size"] > under["metrics"]["fleet_size"]
    assert over["metrics"]["cpu_utilization"] < under["metrics"]["cpu_utilization"]
    client.delete(f"/api/session/{sid}")


@needs_models
def test_fault_injection_recovers_tasks():
    sid = client.post("/api/session", json={"strategy": "full", "initial_fleet": 4}).json()["session_id"]
    client.post(f"/api/session/{sid}/inject",
                json={"task_count": 12, "cpu_per_task": 0.4, "ram_per_task": 0.8})
    state = client.get(f"/api/session/{sid}").json()
    victim = next((vm for vm in state["fleet"] if vm["task_count"] > 0), None)
    assert victim is not None

    result = client.post(f"/api/session/{sid}/fault/{victim['id']}").json()["fault"]
    assert result["ok"] is True
    assert result["tasks_lost"] == 0
    client.delete(f"/api/session/{sid}")


@needs_models
def test_fleet_cap_is_enforced_through_the_api():
    sid = client.post("/api/session", json={"strategy": "full"}).json()["session_id"]
    body = client.post(f"/api/session/{sid}/scale",
                       json={"instance_type": "large", "count": 20}).json()
    assert body["metrics"]["fleet_size"] <= body["config"]["max_fleet"]
    client.delete(f"/api/session/{sid}")
