"""Pydantic request/response models for the public API."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    num_tasks: float = Field(..., ge=0, le=10_000)
    cpu_per_task: float = Field(0.42, gt=0, le=64)
    ram_per_task: float = Field(0.85, gt=0, le=256)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    algo: Literal["xgboost", "rf", "lr"] = "xgboost"
    explain: bool = True


class PredictResponse(BaseModel):
    predicted_cpu: float
    predicted_ram: float
    algo: str
    horizon_intervals: int
    explanation: Optional[dict] = None


class SessionCreateRequest(BaseModel):
    predictor_algo: Literal["xgboost", "rf", "lr"] = "xgboost"
    anomaly_method: Literal["isolation_forest", "zscore"] = "isolation_forest"
    strategy: Literal[
        "static_rules", "threshold_reactive", "ml_predictive",
        "multicloud_only", "q_learning", "rl_only", "full",
    ] = "full"
    initial_fleet: int = Field(3, ge=1, le=20)
    seed: int = 42


class StepRequest(BaseModel):
    ticks: int = Field(1, ge=1, le=288)
    train_rl: bool = True


class InjectRequest(BaseModel):
    task_count: int = Field(10, ge=1, le=500)
    cpu_per_task: float = Field(1.0, gt=0, le=32)
    ram_per_task: float = Field(1.0, gt=0, le=128)
    duration: float = Field(900.0, gt=0, le=86_400)


class ScaleRequest(BaseModel):
    instance_type: Literal["small", "medium", "large"] = "medium"
    count: int = Field(1, ge=1, le=20)
    provider: Optional[Literal["AWS", "Azure", "GCP"]] = None


class ScenarioRequest(BaseModel):
    scenario: Literal["over_provisioned", "under_provisioned", "balanced", "reset"]


class AblationRequest(BaseModel):
    ticks: int = Field(288, ge=24, le=1152)
    seed: int = 42
    strategies: Optional[list[str]] = None


class ProviderQuery(BaseModel):
    instance_type: Literal["small", "medium", "large"] = "medium"
    region: Literal["US-East", "EU-West", "Asia-South"] = "US-East"
    weight_cost: float = Field(0.55, ge=0, le=1)
    weight_latency: float = Field(0.25, ge=0, le=1)
    weight_carbon: float = Field(0.20, ge=0, le=1)


class AnomalyRequest(BaseModel):
    cpu_demand: float = Field(..., ge=0)
    ram_demand: float = Field(..., ge=0)
    cpu_prev: Optional[float] = None
    cpu_rolling: Optional[float] = None
    ram_rolling: Optional[float] = None
    method: Literal["isolation_forest", "zscore"] = "isolation_forest"


class ApiError(BaseModel):
    detail: str
    hint: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    artifacts_ready: bool
    detail: dict[str, Any]
