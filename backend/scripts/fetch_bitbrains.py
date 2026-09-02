"""Build a workload dataset from the **Bitbrains GWA-T-12** production trace.

This replaces the synthetic generator with real telemetry. The trace is one
month (August 2013) of per-VM measurements from a distributed datacentre
operated by Bitbrains, a service provider hosting business-critical enterprise
applications. It is sampled at **five-minute intervals**, which matches this
project's control interval exactly, and records CPU *and* memory - both of which
the allocator needs.

Source
------
Grid Workloads Archive, dataset GWA-T-12 (Shen, Van Beek & Iosup, 2015). The
canonical host (gwa.ewi.tudelft.nl) is frequently unreachable, so this script
reads the mirror maintained by the MUSE research lab:

    https://github.com/muse-research-lab/cloud-forecast-data-persistence

Aggregation
-----------
The optimiser operates on datacentre-level demand, not per-VM series, so the
sampled VMs are summed per timestamp:

    cpu_demand  = sum over VMs of (CPU usage [%] / 100) x (CPU cores)   [cores]
    ram_demand  = sum over VMs of (Memory usage [KB]) / 1048576         [GB]
    num_tasks   = count of VMs with non-zero CPU usage in that interval

**Sampling is documented, not hidden.** The full fastStorage set is 1,241 VMs
(~1.1 GB); this script draws a uniform random sample without replacement under a
fixed seed, so the result is reproducible and the sample size is reported in the
output manifest.

Usage
-----
    python scripts/fetch_bitbrains.py --vms 120 --seed 42
    python scripts/fetch_bitbrains.py --vms 300 --subset 201308
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW = "https://raw.githubusercontent.com/muse-research-lab/cloud-forecast-data-persistence/main"
API = "https://api.github.com/repos/muse-research-lab/cloud-forecast-data-persistence/git/trees/HEAD?recursive=1"

SUBSETS = {
    "1201308": "fastStorage - 1,241 VMs on SAN storage, August 2013",
    "201307": "rnd - 500 VMs, July 2013",
    "201308": "rnd - 500 VMs, August 2013",
    "201309": "rnd - 500 VMs, September 2013",
}

INTERVAL_SECONDS = 300


def _get(url: str, retries: int = 3, timeout: int = 60) -> bytes:
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed after {retries} attempts: {url} ({last})")


def list_vm_files(subset: str) -> list[str]:
    tree = json.loads(_get(API, timeout=120))
    prefix = f"bitbrains/{subset}/"
    return sorted(
        t["path"] for t in tree.get("tree", [])
        if t["type"] == "blob" and t["path"].startswith(prefix) and t["path"].endswith(".csv")
    )


def load_vm(path: str) -> pd.DataFrame | None:
    """Fetch one VM's series and reduce it to (slot, cpu_cores, ram_gb, active)."""
    try:
        raw = _get(f"{RAW}/{path}")
    except RuntimeError:
        return None

    try:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception:
        return None

    df.columns = [str(c).strip().rstrip(";").strip() for c in df.columns]
    need = ["Timestamp [ms]", "CPU cores", "CPU usage [%]", "Memory usage [KB]"]
    if any(c not in df.columns for c in need):
        return None

    ts = pd.to_numeric(df["Timestamp [ms]"], errors="coerce")
    cores = pd.to_numeric(df["CPU cores"], errors="coerce")
    pct = pd.to_numeric(df["CPU usage [%]"], errors="coerce")
    mem = pd.to_numeric(df["Memory usage [KB]"], errors="coerce")

    out = pd.DataFrame({
        # Snap to the 5-minute grid; the raw timestamps drift by a few seconds.
        "slot": (ts // INTERVAL_SECONDS).astype("Int64"),
        "cpu_cores": (pct / 100.0) * cores,
        "ram_gb": mem / 1048576.0,
    }).dropna()

    out["active"] = (out["cpu_cores"] > 1e-6).astype(int)
    # A VM occasionally reports twice in one slot; take the mean rather than
    # double-counting its demand.
    return out.groupby("slot", as_index=False).agg(
        cpu_cores=("cpu_cores", "mean"),
        ram_gb=("ram_gb", "mean"),
        active=("active", "max"),
    )


def label_bursts(cpu: np.ndarray, k: float = 6.0, refractory: int = 6) -> np.ndarray:
    """Heuristic burst-onset labels for a real trace.

    Real telemetry carries no ground-truth anomaly annotation. Onsets are
    therefore marked where the first difference exceeds `k` times the median
    absolute deviation of the first difference - a robust, scale-free rule that
    does not depend on any model being evaluated. A refractory period prevents
    one event being counted several times as it develops.

    These are *heuristic* labels. Detection scores computed against them measure
    agreement with an independent statistical rule, not with a known truth, and
    must be reported as such.
    """
    d = np.diff(cpu, prepend=cpu[0])
    mad = np.median(np.abs(d - np.median(d))) or 1e-9
    flags = d > (np.median(d) + k * 1.4826 * mad)

    onsets = np.zeros(len(cpu), dtype=int)
    cooldown = 0
    for i, f in enumerate(flags):
        if cooldown > 0:
            cooldown -= 1
            continue
        if f:
            onsets[i] = 1
            cooldown = refractory
    return onsets


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vms", type=int, default=120, help="VMs to sample (0 = all)")
    ap.add_argument("--subset", default="1201308", choices=list(SUBSETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "workload_bitbrains.csv")
    args = ap.parse_args()

    print(f"Bitbrains GWA-T-12  subset={args.subset}  ({SUBSETS[args.subset]})")
    print("Listing VM files...")
    files = list_vm_files(args.subset)
    if not files:
        print("ERROR: no files found for that subset.", file=sys.stderr)
        return 1
    print(f"  {len(files)} VMs available")

    chosen = files
    if args.vms and args.vms < len(files):
        chosen = random.Random(args.seed).sample(files, args.vms)
        print(f"  sampling {len(chosen)} uniformly at random (seed {args.seed})")

    cpu_by_slot: dict[int, float] = defaultdict(float)
    ram_by_slot: dict[int, float] = defaultdict(float)
    act_by_slot: dict[int, int] = defaultdict(int)

    ok = skipped = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(load_vm, p): p for p in chosen}
        for n, fut in enumerate(as_completed(futures), start=1):
            df = fut.result()
            if df is None or df.empty:
                skipped += 1
            else:
                ok += 1
                for slot, c, r, a in df.itertuples(index=False):
                    cpu_by_slot[slot] += c
                    ram_by_slot[slot] += r
                    act_by_slot[slot] += a
            if n % 20 == 0 or n == len(chosen):
                print(f"  {n}/{len(chosen)} fetched  ({ok} ok, {skipped} skipped)", flush=True)

    if not cpu_by_slot:
        print("ERROR: no usable rows.", file=sys.stderr)
        return 1

    slots = np.array(sorted(cpu_by_slot))
    # Keep the longest run of consecutive slots: a gap would silently become a
    # discontinuity in every lag feature downstream.
    breaks = np.flatnonzero(np.diff(slots) != 1)
    bounds = np.concatenate(([0], breaks + 1, [len(slots)]))
    spans = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    lo, hi = max(spans, key=lambda s: s[1] - s[0])
    if (hi - lo) < len(slots):
        print(f"  trimmed to the longest contiguous run: {hi - lo} of {len(slots)} slots")
    slots = slots[lo:hi]

    ts = pd.to_datetime(slots * INTERVAL_SECONDS, unit="s")
    cpu = np.array([cpu_by_slot[s] for s in slots], dtype=float)
    ram = np.array([ram_by_slot[s] for s in slots], dtype=float)
    active = np.array([max(1, act_by_slot[s]) for s in slots], dtype=float)

    df = pd.DataFrame({
        "timestamp": ts,
        "num_tasks": active,
        "cpu_per_task": cpu / active,
        "ram_per_task": ram / active,
        "hour": ts.hour,
        "day_of_week": ts.dayofweek,
        "cpu_demand": cpu,
        "ram_demand": ram,
        "is_weekend": (ts.dayofweek >= 5).astype(int),
        "burst_onset": label_bursts(cpu),
        "interval": np.arange(len(slots)),
    })
    df["burst_active"] = df["burst_onset"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    manifest = {
        "source": "Bitbrains GWA-T-12 (Grid Workloads Archive)",
        "mirror": "github.com/muse-research-lab/cloud-forecast-data-persistence",
        "subset": args.subset,
        "subset_description": SUBSETS[args.subset],
        "vms_available": len(files),
        "vms_sampled": len(chosen),
        "vms_loaded": ok,
        "vms_skipped": skipped,
        "sampling": "uniform without replacement" if len(chosen) < len(files) else "all",
        "seed": args.seed,
        "interval_seconds": INTERVAL_SECONDS,
        "rows": len(df),
        "span_start": str(df["timestamp"].iloc[0]),
        "span_end": str(df["timestamp"].iloc[-1]),
        "burst_labels": "heuristic (MAD rule on first difference, k=6, refractory=6)",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (args.out.parent / "workload_bitbrains_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {len(df):,} rows to {args.out}")
    print(f"  span        : {manifest['span_start']}  ->  {manifest['span_end']}")
    print(f"  VMs         : {ok} loaded, {skipped} skipped, of {len(files)} available")
    print(f"  cpu_demand  : mean {cpu.mean():7.2f}  min {cpu.min():6.2f}  max {cpu.max():7.2f} cores")
    print(f"  ram_demand  : mean {ram.mean():7.2f}  min {ram.min():6.2f}  max {ram.max():7.2f} GB")
    print(f"  RAM:CPU     : {ram.mean() / max(cpu.mean(), 1e-9):.2f}")
    print(f"  burst labels: {int(df['burst_onset'].sum())} onsets "
          f"({df['burst_onset'].mean() * 100:.2f}% of intervals)")
    print(f"  elapsed     : {manifest['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
