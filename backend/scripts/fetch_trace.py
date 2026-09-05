"""Build a workload dataset from a **real production trace**.

Supports four public cloud traces, so a finding can be tested across workload
classes rather than asserted from one dataset:

    bitbrains   Enterprise VMs, Bitbrains datacentre, Aug-Sep 2013 (GWA-T-12)
    google      Google Borg cluster tasks, 2019
    azure       Microsoft Azure VM CPU readings, 2017/2019
    alibaba     Alibaba production cluster machines, 2018

All four are read from the MUSE research lab mirror, which republishes them in a
consistent per-entity CSV layout:

    https://github.com/muse-research-lab/cloud-forecast-data-persistence

Aggregation
-----------
The optimiser controls datacentre-level demand, not per-entity series, so the
sampled entities are summed per five-minute slot:

    cpu_demand = sum of per-entity CPU, in cores
    ram_demand = sum of per-entity memory, in GB
    num_tasks  = count of entities active in that slot

Unit normalisation is per-dataset and documented in `ADAPTERS` below. Where a
trace reports normalised utilisation rather than absolute resources, it is
scaled by a nominal machine size; the scale factor is recorded in the manifest.
**Scaling is a linear transform, so it does not affect R², rank correlation or
any relative comparison** - it only puts the series in units the allocator's
instance catalogue can serve.

Usage
-----
    python scripts/fetch_trace.py --dataset bitbrains --entities 300
    python scripts/fetch_trace.py --dataset google --entities 200
    python scripts/fetch_trace.py --dataset alibaba --entities 400
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
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW = "https://raw.githubusercontent.com/muse-research-lab/cloud-forecast-data-persistence/main"
API = "https://api.github.com/repos/muse-research-lab/cloud-forecast-data-persistence/git/trees/HEAD?recursive=1"

SLOT_SECONDS = 300


# ---------------------------------------------------------------------------
# Dataset adapters
# ---------------------------------------------------------------------------

@dataclass
class Adapter:
    prefix: str
    description: str
    parse: Callable[[pd.DataFrame], pd.DataFrame | None]
    notes: str = ""
    ram_is_synthetic: bool = False
    scale: dict = field(default_factory=dict)


def _frame(slot, cpu, ram) -> pd.DataFrame:
    out = pd.DataFrame({"slot": slot, "cpu_cores": cpu, "ram_gb": ram}).dropna()
    out = out[out["slot"] >= 0]
    out["active"] = (out["cpu_cores"] > 1e-6).astype(int)
    # An entity can report twice inside one slot; take the mean rather than
    # double-counting its demand.
    return out.groupby("slot", as_index=False).agg(
        cpu_cores=("cpu_cores", "mean"),
        ram_gb=("ram_gb", "mean"),
        active=("active", "max"),
    )


def _parse_bitbrains(df: pd.DataFrame) -> pd.DataFrame | None:
    df.columns = [str(c).strip().rstrip(";").strip() for c in df.columns]
    need = ["Timestamp [ms]", "CPU cores", "CPU usage [%]", "Memory usage [KB]"]
    if any(c not in df.columns for c in need):
        return None
    ts = pd.to_numeric(df["Timestamp [ms]"], errors="coerce")
    cores = pd.to_numeric(df["CPU cores"], errors="coerce")
    pct = pd.to_numeric(df["CPU usage [%]"], errors="coerce")
    mem = pd.to_numeric(df["Memory usage [KB]"], errors="coerce")
    # Absolute units already: percentage of the VM's own cores, and memory in KB.
    return _frame(ts // SLOT_SECONDS, (pct / 100.0) * cores, mem / 1048576.0)


# Google normalises usage against the largest machine in the cell, so a value of
# 1.0 means "a whole machine". A nominal machine size converts that back into
# cores and GB the allocator can reason about.
GOOGLE_MACHINE_CORES = 64.0
GOOGLE_MACHINE_RAM_GB = 256.0


def _parse_google(df: pd.DataFrame) -> pd.DataFrame | None:
    if "start_time" not in df.columns or "avg_cpu_usage" not in df.columns:
        return None
    # start_time is in microseconds since the trace epoch.
    ts = pd.to_numeric(df["start_time"], errors="coerce") / 1_000_000.0
    cpu = pd.to_numeric(df["avg_cpu_usage"], errors="coerce") * GOOGLE_MACHINE_CORES
    mem = pd.to_numeric(df.get("avg_mem_usage"), errors="coerce") * GOOGLE_MACHINE_RAM_GB
    return _frame(ts // SLOT_SECONDS, cpu, mem)


# Azure publishes CPU only. RAM is therefore derived from a fixed ratio and
# flagged as synthetic in the manifest - it must not be used for any
# memory-specific claim. CPU-only experiments (the horizon study) are unaffected.
AZURE_VM_CORES = 2.0
AZURE_RAM_PER_CORE_GB = 2.0


def _parse_azure(df: pd.DataFrame) -> pd.DataFrame | None:
    if "timestamp" not in df.columns or "avg_cpu" not in df.columns:
        return None
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    cpu = pd.to_numeric(df["avg_cpu"], errors="coerce") / 100.0 * AZURE_VM_CORES
    return _frame(ts // SLOT_SECONDS, cpu, cpu * AZURE_RAM_PER_CORE_GB)


# Alibaba reports machine-level utilisation at 60-second resolution.
#
# Despite the column names, `cpu_util_percent` and `mem_util_percent` in this
# mirror are **fractions in [0, 1], not percentages**. Verified over 40 randomly
# sampled machine files: no value exceeds 1.0 in either column, and mean CPU is
# 0.395 - which matches the ~40% mean utilisation Alibaba reports for this
# cluster, whereas reading them as percentages would imply 0.4%. Dividing by 100
# here would understate Alibaba demand by two orders of magnitude.
ALIBABA_MACHINE_CORES = 96.0
ALIBABA_MACHINE_RAM_GB = 384.0


def _parse_alibaba(df: pd.DataFrame) -> pd.DataFrame | None:
    if "time_stamp" not in df.columns or "cpu_util_percent" not in df.columns:
        return None
    ts = pd.to_numeric(df["time_stamp"], errors="coerce")
    cpu = pd.to_numeric(df["cpu_util_percent"], errors="coerce")
    mem = pd.to_numeric(df.get("mem_util_percent"), errors="coerce")
    if float(pd.concat([cpu, mem]).max(skipna=True) or 0) > 1.5:
        # A future revision of the mirror may switch to true percentages; detect
        # rather than silently producing a 100x error either way.
        cpu, mem = cpu / 100.0, mem / 100.0
    # 60 s -> 300 s happens naturally in the slot groupby.
    return _frame(ts // SLOT_SECONDS,
                  cpu * ALIBABA_MACHINE_CORES, mem * ALIBABA_MACHINE_RAM_GB)


ADAPTERS: dict[str, Adapter] = {
    "bitbrains": Adapter(
        "bitbrains/1201308/", "Bitbrains GWA-T-12 fastStorage - enterprise VMs, Aug 2013",
        _parse_bitbrains,
        notes="Absolute units: CPU usage % of the VM's own cores; memory in KB.",
    ),
    "google": Adapter(
        "google/", "Google Borg cluster tasks, 2019",
        _parse_google,
        notes=f"Usage normalised to the largest machine; scaled by a nominal "
              f"{GOOGLE_MACHINE_CORES:.0f}-core / {GOOGLE_MACHINE_RAM_GB:.0f} GB machine.",
        scale={"machine_cores": GOOGLE_MACHINE_CORES, "machine_ram_gb": GOOGLE_MACHINE_RAM_GB},
    ),
    "azure": Adapter(
        "azure/", "Microsoft Azure VM CPU readings",
        _parse_azure,
        notes=f"CPU only in the source. Scaled by a nominal {AZURE_VM_CORES:.0f}-vCPU VM; "
              f"RAM is DERIVED at {AZURE_RAM_PER_CORE_GB:.0f} GB/core and is not real.",
        ram_is_synthetic=True,
        scale={"vm_cores": AZURE_VM_CORES, "ram_per_core_gb": AZURE_RAM_PER_CORE_GB},
    ),
    "alibaba": Adapter(
        "alibaba/", "Alibaba production cluster machines, 2018",
        _parse_alibaba,
        notes=f"Machine utilisation *fractions* at 60 s (the '_percent' column names are a "
              f"misnomer in this mirror), aggregated to 300 s; scaled by a nominal "
              f"{ALIBABA_MACHINE_CORES:.0f}-core / {ALIBABA_MACHINE_RAM_GB:.0f} GB machine.",
        scale={"machine_cores": ALIBABA_MACHINE_CORES, "machine_ram_gb": ALIBABA_MACHINE_RAM_GB},
    ),
}


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

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


def list_files(prefix: str) -> list[str]:
    tree = json.loads(_get(API, timeout=180))
    return sorted(
        t["path"] for t in tree.get("tree", [])
        if t["type"] == "blob" and t["path"].startswith(prefix) and t["path"].endswith(".csv")
    )


def load_entity(path: str, adapter: Adapter) -> pd.DataFrame | None:
    try:
        raw = _get(f"{RAW}/{path}")
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")
    except Exception:
        return None
    try:
        return adapter.parse(df)
    except Exception:
        return None


def label_bursts(cpu: np.ndarray, k: float = 6.0, refractory: int = 6) -> np.ndarray:
    """Heuristic burst-onset labels for a real trace.

    Real telemetry carries no ground-truth anomaly annotation. Onsets are marked
    where the first difference exceeds `k` times the median absolute deviation of
    the first difference - a robust, scale-free rule independent of any model
    being evaluated. Detection scores against these labels measure agreement with
    a statistical rule, not with known truth, and must be reported as such.
    """
    if len(cpu) < 3:
        return np.zeros(len(cpu), dtype=int)
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
    ap.add_argument("--dataset", default="bitbrains", choices=list(ADAPTERS))
    ap.add_argument("--entities", type=int, default=300, help="VMs/tasks to sample (0 = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--collection", default=None,
                    help="sub-collection to read instead of the adapter default. "
                         "Bitbrains publishes four: 1201308 (fastStorage, 1241 VMs) "
                         "and 201307 / 201308 / 201309 (Rnd, 500 VMs each, different "
                         "months and different VMs).")
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="take partition K of N from the entity list, so repeated "
                         "runs give provably DISJOINT entity samples. Two seeds do "
                         "not: independent draws from the same pool overlap, and a "
                         "study that treated them as separate workloads would be "
                         "counting the same VMs twice.")
    ap.add_argument("--name", default=None,
                    help="output basename (default: the dataset name)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min-coverage", type=float, default=0.80,
                    help="a slot counts as measured only if at least this fraction of "
                         "the p95 entity count reports in it, so the aggregate is not "
                         "an artefact of entities entering and leaving the window")
    ap.add_argument("--max-gap", type=int, default=6,
                    help="interpolate runs of low-coverage slots up to this length "
                         "(6 slots = 30 min); a longer outage splits the series")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    adapter = ADAPTERS[args.dataset]
    if args.collection:
        root = adapter.prefix.split("/")[0]
        adapter = replace(adapter, prefix=f"{root}/{args.collection}/")
    name = args.name or args.dataset
    out = args.out or (ROOT / "data" / f"workload_{name}.csv")

    print(f"{args.dataset}: {adapter.description}")
    if adapter.notes:
        print(f"  {adapter.notes}")
    if adapter.ram_is_synthetic:
        print("  WARNING: memory is derived, not measured. CPU-only claims are valid.")

    print("Listing entity files...")
    files = list_files(adapter.prefix)
    if not files:
        print(f"ERROR: no files under {adapter.prefix}", file=sys.stderr)
        return 1
    print(f"  {len(files)} entities available")

    # Shuffle once with the seed, then either take a prefix or a disjoint
    # partition. Partitioning a single shuffled order is what makes the shards
    # provably non-overlapping.
    pool = list(files)
    random.Random(args.seed).shuffle(pool)
    shard_label = None
    if args.shard:
        k, n = (int(x) for x in args.shard.split("/"))
        if not (0 <= k < n):
            print(f"ERROR: --shard {args.shard} must satisfy 0 <= K < N", file=sys.stderr)
            return 1
        pool = pool[k::n]
        shard_label = f"{k}/{n}"
        print(f"  shard {shard_label}: {len(pool)} of {len(files)} entities, "
              f"disjoint from the other {n - 1} shards")

    chosen = pool
    if args.entities and args.entities < len(pool):
        chosen = pool[:args.entities]
        print(f"  sampling {len(chosen)} uniformly at random (seed {args.seed})")

    cpu_by: dict[int, float] = defaultdict(float)
    ram_by: dict[int, float] = defaultdict(float)
    act_by: dict[int, int] = defaultdict(int)
    seen_by: dict[int, int] = defaultdict(int)   # entities reporting in this slot
    ok = skipped = 0
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(load_entity, p, adapter): p for p in chosen}
        for n, fut in enumerate(as_completed(futures), start=1):
            df = fut.result()
            if df is None or df.empty:
                skipped += 1
            else:
                ok += 1
                for slot, c, r, a in df.itertuples(index=False):
                    cpu_by[slot] += c
                    ram_by[slot] += r
                    act_by[slot] += a
                    seen_by[slot] += 1
            if n % 50 == 0 or n == len(chosen):
                print(f"  {n}/{len(chosen)} fetched ({ok} ok, {skipped} skipped)", flush=True)

    if not cpu_by:
        print("ERROR: no usable rows.", file=sys.stderr)
        return 1

    all_slots = np.array(sorted(cpu_by))
    coverage = np.array([seen_by[s] for s in all_slots], dtype=float)
    peak_cover = float(np.percentile(coverage, 95))
    usable = coverage >= args.min_coverage * peak_cover
    if not usable.any():
        print("ERROR: no slot meets the coverage threshold.", file=sys.stderr)
        return 1

    # Two different problems, which need two different treatments.
    #
    # 1. Entities do not all span the same wall-clock window, so the aggregate
    #    ramps up at the start and down at the end purely as an artefact of how
    #    many entities were reporting. Those edges are trimmed away.
    # 2. Inside the trimmed window, isolated slots can still dip below the
    #    threshold. Treating each dip as a hard break is disastrous: on Bitbrains
    #    coverage is a healthy 279-294 throughout, but a handful of scattered
    #    dips fragment the trace so badly that the longest gap-free run is 2,972
    #    of 8,640 slots. Short dips are interpolated instead, and only a sustained
    #    outage (> --max-gap slots) actually splits the series.
    live = np.flatnonzero(usable)
    first_slot = int(all_slots[live[0]])
    last_slot = int(all_slots[live[-1]])
    trimmed_edges = len(all_slots) - (live[-1] - live[0] + 1)

    good = {int(s): (cpu_by[s], ram_by[s], act_by[s])
            for s, keep in zip(all_slots, usable) if keep}

    # Walk the full integer slot range and split it only on sustained outages.
    full = np.arange(first_slot, last_slot + 1)
    present = np.array([int(s) in good for s in full])
    segments, start, gap_start = [], 0, None
    for i, ok_here in enumerate(present):
        if ok_here:
            if gap_start is not None and (i - gap_start) > args.max_gap:
                segments.append((start, gap_start))
                start = i
            gap_start = None
        elif gap_start is None:
            gap_start = i
    segments.append((start, len(present) if gap_start is None else gap_start))

    lo, hi = max(segments, key=lambda s: s[1] - s[0])
    slots = full[lo:hi]
    repaired = int((~present[lo:hi]).sum())

    print(f"  coverage: {int(coverage.min())}-{int(coverage.max())} entities/slot "
          f"(p95 {peak_cover:.0f}, threshold {args.min_coverage:.0%})")
    print(f"  kept {len(slots):,} of {len(all_slots):,} slots "
          f"({trimmed_edges:,} trimmed at the edges, {repaired:,} interpolated over "
          f"gaps of <= {args.max_gap} slots)")

    if len(slots) < 500:
        print(f"ERROR: only {len(slots)} usable contiguous slots; too short to evaluate.",
              file=sys.stderr)
        return 1
    if repaired > 0.05 * len(slots):
        print(f"WARNING: {repaired / len(slots):.1%} of the kept series is "
              f"interpolated rather than measured.", file=sys.stderr)

    # Anchor to a wall-clock so the hour / day-of-week features have something to
    # key on. Bitbrains timestamps are true Unix time; the other three are
    # relative to their own trace epoch, so their anchor is a fixed offset and
    # the *phase* of "hour 14" is arbitrary. The periodicity a model can learn is
    # unaffected - only the label is - and the manifest records which case applies.
    real_clock = args.dataset == "bitbrains"
    origin = (pd.Timestamp("1970-01-01") + pd.Timedelta(seconds=int(slots[0]) * SLOT_SECONDS)
              if real_clock else pd.Timestamp("2013-08-12"))
    ts = pd.to_datetime((slots - slots[0]) * SLOT_SECONDS, unit="s", origin=origin)

    # Slots that failed the coverage check carry NaN and are then linearly
    # interpolated. Carrying the low-coverage reading forward instead would inject
    # a fake dip in demand; interpolating keeps the series continuous, which every
    # lag and rolling feature downstream depends on.
    def series(index: int, minimum: float = 0.0) -> np.ndarray:
        raw = np.array([good[int(s)][index] if int(s) in good else np.nan
                        for s in slots], dtype=float)
        filled = pd.Series(raw).interpolate(limit_direction="both").to_numpy()
        return np.maximum(filled, minimum)

    cpu = series(0)
    ram = series(1)
    active = series(2, minimum=1.0)

    frame = pd.DataFrame({
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
    frame["burst_active"] = frame["burst_onset"]

    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)

    manifest = {
        "dataset": args.dataset,
        "name": name,
        "collection": args.collection or adapter.prefix,
        "shard": shard_label,
        "description": adapter.description,
        "mirror": "github.com/muse-research-lab/cloud-forecast-data-persistence",
        "unit_notes": adapter.notes,
        "ram_is_synthetic": adapter.ram_is_synthetic,
        "scale_factors": adapter.scale,
        "entities_available": len(files),
        "entities_sampled": len(chosen),
        "entities_loaded": ok,
        "entities_skipped": skipped,
        "sampling": "uniform without replacement" if len(chosen) < len(files) else "all",
        "seed": args.seed,
        "slot_seconds": SLOT_SECONDS,
        "min_coverage": args.min_coverage,
        "max_gap_interpolated": args.max_gap,
        "slots_before_coverage_filter": len(all_slots),
        "slots_trimmed_at_edges": int(trimmed_edges),
        "slots_interpolated": repaired,
        "entities_per_slot_p95": round(peak_cover, 1),
        "entities_per_slot_min": int(coverage.min()),
        "wallclock_anchor": ("true Unix timestamps from the source trace" if real_clock
                             else "arbitrary fixed offset - source timestamps are "
                                  "relative to the trace epoch, so hour/day-of-week "
                                  "labels carry a constant unknown phase shift"),
        "span_days": round(len(slots) * SLOT_SECONDS / 86400.0, 2),
        "rows": len(frame),
        "cpu_mean": round(float(cpu.mean()), 3),
        "cpu_max": round(float(cpu.max()), 3),
        "ram_mean": round(float(ram.mean()), 3),
        "ram_cpu_ratio": round(float(ram.mean() / max(cpu.mean(), 1e-9)), 3),
        "burst_onsets": int(frame["burst_onset"].sum()),
        "burst_labels": "heuristic (MAD rule on first difference, k=6, refractory=6)",
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out.parent / f"workload_{name}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {len(frame):,} rows to {out}")
    print(f"  entities   : {ok} loaded, {skipped} skipped, of {len(files)} available")
    print(f"  cpu_demand : mean {cpu.mean():8.2f}  max {cpu.max():8.2f} cores")
    print(f"  ram_demand : mean {ram.mean():8.2f}  max {ram.max():8.2f} GB"
          f"{'  (DERIVED)' if adapter.ram_is_synthetic else ''}")
    print(f"  RAM:CPU    : {manifest['ram_cpu_ratio']}")
    print(f"  bursts     : {manifest['burst_onsets']} onsets "
          f"({frame['burst_onset'].mean() * 100:.2f}%)")
    print(f"  elapsed    : {manifest['elapsed_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
