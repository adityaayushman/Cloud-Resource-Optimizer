"""In-memory simulation sessions.

Each browser gets its own fleet so two people using the deployed dashboard do
not mutate one another's cluster. Sessions expire on a TTL because the process
is long-lived and abandoned sessions would otherwise leak.

This is deliberately not a database: the fleet is an ephemeral simulation, and
persisting it would add a dependency the project does not otherwise need. The
consequence - state is lost on a Render cold start - is documented in the
README rather than hidden.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from .engine import SmartAllocator
from .models import CloudProvider, InstanceType
from .workload import WorkloadConfig, WorkloadGenerator

SESSION_TTL_SECONDS = 60 * 45
MAX_SESSIONS = 200
MAX_LOG_LINES = 200


@dataclass
class Session:
    id: str
    allocator: SmartAllocator
    generator: WorkloadGenerator
    strategy: str
    created_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    tick_index: int = 0
    # CPU consumed in the previous interval. A reactive autoscaler must act on
    # this, not on instantaneous usage: at the top of a tick the previous
    # cohort has already retired and the fleet reads as idle.
    last_observed_cpu: float | None = None
    logs: list[dict] = field(default_factory=list)
    history: list[dict] = field(default_factory=list)
    autoscaler: object | None = None

    def log(self, source: str, message: str) -> None:
        self.logs.append({
            "t": round(self.allocator.clock, 1),
            "source": source,
            "message": message,
        })
        if len(self.logs) > MAX_LOG_LINES:
            del self.logs[: len(self.logs) - MAX_LOG_LINES]

    def touch(self) -> None:
        self.last_seen = time.time()


class SessionStore:
    def __init__(self, artifacts_dir=None):
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()
        self.artifacts_dir = artifacts_dir

    # -- lifecycle -------------------------------------------------------

    def create(
        self,
        predictor_algo: str = "xgboost",
        anomaly_method: str = "isolation_forest",
        strategy: str = "full",
        initial_fleet: int = 3,
        seed: int = 42,
    ) -> Session:
        self.purge_expired()
        with self._lock:
            if len(self._sessions) >= MAX_SESSIONS:
                oldest = min(self._sessions.values(), key=lambda s: s.last_seen)
                self._sessions.pop(oldest.id, None)

        multi_cloud = strategy in ("multicloud_only", "full")
        allocator = SmartAllocator(
            predictor_algo=predictor_algo,
            anomaly_method=anomaly_method,
            artifacts_dir=self.artifacts_dir,
            multi_cloud=multi_cloud,
            seed=seed,
        )
        allocator.dqn_agent.load(allocator.artifacts_dir / "dqn_agent.json")

        for _ in range(initial_fleet):
            allocator.add_vm(
                InstanceType.MEDIUM,
                provider=None if multi_cloud else CloudProvider.AWS,
            )

        session = Session(
            id=uuid.uuid4().hex[:16],
            allocator=allocator,
            generator=WorkloadGenerator(WorkloadConfig(interval_minutes=5), seed=seed),
            strategy=strategy,
        )

        if strategy == "threshold_reactive":
            from .engine import AutoScaler

            session.autoscaler = AutoScaler(allocator, mode="reactive")
        elif strategy in ("ml_predictive", "multicloud_only"):
            from .engine import AutoScaler

            session.autoscaler = AutoScaler(allocator, mode="predictive")

        session.log("INIT", f"Session created - strategy={strategy}, "
                            f"predictor={predictor_algo}, fleet={initial_fleet}")
        if allocator.load_errors:
            for err in allocator.load_errors:
                session.log("WARN", f"Artifact unavailable: {err}")

        with self._lock:
            self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            session = self._sessions.get(session_id)
        if session:
            session.touch()
        return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def purge_expired(self) -> int:
        cutoff = time.time() - SESSION_TTL_SECONDS
        with self._lock:
            stale = [sid for sid, s in self._sessions.items() if s.last_seen < cutoff]
            for sid in stale:
                self._sessions.pop(sid, None)
        return len(stale)

    def stats(self) -> dict:
        with self._lock:
            return {
                "active_sessions": len(self._sessions),
                "max_sessions": MAX_SESSIONS,
                "ttl_seconds": SESSION_TTL_SECONDS,
            }
