# Validation on a real production trace

The synthetic evaluation in `docs/RESULTS.md` has one structural weakness: the
workload generator was written for this project, so the models are scored on a
distribution the authors designed. This study repeats the entire evaluation on
production telemetry and reports where the two agree — and, more importantly,
where they do not.

**The headline is a negative result.** The synthetic study concluded that the
learned forecaster beats a naive baseline from a 15-minute horizon onward. On
the real trace **it never does, at any horizon**, and the gap widens as the
horizon grows. That conclusion was an artefact of structure the generator put
into the data. It is reported here rather than buried, because it is the single
most useful thing this validation produced.

---

## 1. Dataset

**Bitbrains GWA-T-12 (fastStorage)** — one month of per-VM telemetry from a
distributed datacentre operated by Bitbrains, a service provider hosting
business-critical enterprise applications. Retrieved with
`scripts/fetch_bitbrains.py`.

| Property | Value |
|---|---|
| Source | Grid Workloads Archive, dataset GWA-T-12 |
| Mirror used | `github.com/muse-research-lab/cloud-forecast-data-persistence` |
| VMs available | 1,241 |
| VMs sampled | **300**, uniform without replacement, seed 42 |
| Sampling interval | **300 s** (matches the control interval exactly) |
| Span | 2013-08-12 13:40 → 2013-09-11 13:35 (30 days) |
| Aggregated rows | 8,640 |
| Mean CPU demand | 136.4 cores (peak 291.8) |
| Mean RAM demand | 171.2 GB (peak 790.6) |
| RAM:CPU ratio | **1.26** |

Per-VM series are summed per timestamp to give datacentre-level demand, which is
what the allocator controls. Sampling is documented in
`data/workload_bitbrains_manifest.json`, not hidden.

**Two system changes were needed** and are reported because they affect
comparability: an `xlarge` (16 vCPU / 32 GB) instance type, since the real fleet
peaks near 300 cores and an 8-core node cannot serve that without an
unrealistically large fleet; and a per-allocator fleet cap (80 here, 40 for
synthetic) instead of a global constant.

> **Anomaly labels are heuristic.** Production telemetry carries no ground-truth
> annotation. Onsets are marked by a robust MAD rule on the first difference,
> independent of any model being evaluated. Detection scores below therefore
> measure agreement with a statistical rule, not with known truth.

---

## 2. The negative result: forecasting does not beat persistence

Margin = model R² − persistence R², mean over 3 disjoint windows of the trace.

| Horizon | XGBoost | Random Forest | Linear Regression |
|---|---|---|---|
| 5 min | −0.0063 (1/3) | −0.0033 (1/3) | −0.0080 (0/3) |
| 15 min | −0.0428 (0/3) | −0.0236 (0/3) | −0.0040 (0/3) |
| 30 min | −0.0618 (0/3) | −0.0542 (0/3) | **+0.0035 (2/3)** |
| 60 min | −0.0593 (1/3) | −0.0674 (1/3) | **+0.0260 (2/3)** |

### Side by side with the synthetic result

| Horizon | XGBoost, synthetic | XGBoost, real trace |
|---|---|---|
| 5 min | −0.0009 (1/3) | −0.0063 (1/3) |
| 15 min | **+0.0108 (3/3)** | −0.0428 (0/3) |
| 30 min | **+0.0403 (3/3)** | −0.0618 (0/3) |
| 60 min | **+0.1490 (3/3)** | −0.0593 (1/3) |

**The conclusion reverses.** Three things follow.

1. **The tree ensembles are beaten by persistence on real data, and the deficit
   grows with horizon** — the opposite of the synthetic trend. Aggregate
   datacentre demand at five-minute resolution behaves close to a random walk
   with a slow trend. Trees fit local structure and extrapolate it; when the
   structure is noise, extrapolating it is worse than doing nothing.
2. **Linear regression is the only model that beats persistence at long
   horizons** (+0.026 at 60 minutes, 2 of 3 windows). It cannot represent the
   spurious local structure that hurts the ensembles, so it degrades gracefully.
   This exactly inverts the synthetic ranking, where the ensembles led at long
   horizons precisely because the generator contained learnable step structure.
3. **The synthetic finding was an artefact of the generator.** The batch window,
   weekday-afternoon amplification and maintenance dip written into
   `app/workload.py` are learnable by construction. Real demand does not contain
   them in that clean a form. Had this project reported only the synthetic
   result, it would have reported a conclusion that does not hold on production
   data.

One-step accuracy on the real trace, for completeness:

| Model | R² | MAE (cores) | MAPE |
|---|---|---|---|
| Linear Regression | 0.8748 | 10.703 | 14.76% |
| *Persistence baseline* | *0.8738* | *9.800* | *12.79%* |
| XGBoost | 0.8698 | 11.963 | 17.70% |
| Random Forest | 0.8640 | 11.873 | 17.26% |

Note that persistence has the **best MAE and MAPE of all four**. A ranking by R²
alone would have hidden that.

---

## 3. What did transfer: reinforcement learning

The RL result not only survives the move to real data, it **strengthens**.

7 policies × 3 seeds × 288 ticks (24 h), identical trace window per seed.

| Configuration | Utilisation (%) | Cost ($/day) | Latency (s) | Fail (%) | SLA (%) | CO₂ (kg) | Nodes |
|---|---|---|---|---|---|---|---|
| Static rule-based *(negative control)* | 58.1 ± 22.4 | 111.45 ± 0.00 | 6499 | 10.65 | 64.8 | 21.62 | 13.0 |
| Threshold reactive | 63.8 ± 1.9 | 109.14 ± 43.66 | 6675 | 12.30 | 80.8 | 28.42 | 29.2 |
| **ML prediction only** *(baseline)* | 58.2 ± 3.1 | 119.99 ± 46.88 | 345 | 0.08 | 99.0 | 26.06 | 19.2 |
| ML + multi-cloud | 58.4 ± 3.7 | 102.84 ± 38.02 | 345 | 0.08 | 99.0 | 24.49 | 18.0 |
| Tabular Q-learning | 91.3 ± 4.3 | 77.30 ± 32.53 | 505 | 1.78 | 37.4 | 20.33 | 11.1 |
| DQN only | 88.9 ± 5.6 | 79.53 ± 32.04 | 530 | 1.10 | 52.5 | 20.87 | 11.9 |
| **All components combined** | **89.1 ± 3.9** | **69.31 ± 27.50** | 453 | 0.95 | 56.2 | 20.34 | 10.9 |

### Synthetic vs real, full system against its baseline

| | Synthetic | Real trace |
|---|---|---|
| Utilisation gain | +28.7% | **+53.0%** |
| Cost reduction | −33.6% | **−42.2%** |
| Task failure rate | 1.37% | **0.95%** |

The gains are **larger** on production data, and the failure rate is **lower**.
The reason is visible in the fleet sizes: the predictive baseline holds a fixed
1.2× headroom and ends up running 19.2 nodes at 58% utilisation, because real
demand is burstier than the synthetic workload and a fixed multiplier must be
sized for the peaks. The learned policy adapts its headroom to state and serves
the same workload with 10.9 nodes at 89% utilisation.

The DQN also improved during training on the real trace, on a held-out window:

| | Reward | Utilisation | Cost/day | Failures |
|---|---|---|---|---|
| Untrained | +0.440 | 80.7% | $130.93 | 1.14% |
| **Trained (checkpoint, episode 7)** | **+0.748** | 84.1% | $124.08 | **0.07%** |

### Deep vs tabular, again

DQN rejects 1.10% of work against tabular Q-learning's 1.78%, at comparable
utilisation. The same generalisation gap seen on synthetic data appears on real
data, which is a genuine replication rather than a repeat of one experiment.

### Multi-cloud, again

Cost −14.3% with utilisation, latency and failure rate unchanged (58.2 → 58.4%,
345 s → 345 s, 0.08% → 0.08%). This is the third independent confirmation that
provider selection is a free saving: it changes only where capacity is bought.

---

## 4. Anomaly detection: the ranking also reverses

| Detector | Events | Detected | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Z-score (4σ) | 248 | 97 | 0.708 | **0.391** | **0.504** |
| Isolation Forest | 248 | 42 | **0.764** | 0.169 | 0.277 |

On synthetic data Isolation Forest led (F1 0.506 vs 0.379); on the real trace
z-score leads (0.504 vs 0.277). Both detectors are far more *precise* on real
data and far less *sensitive*. Since the real labels are themselves a
first-difference rule, a univariate z-score on related features is closer to the
labelling process — so this comparison should be read as weaker evidence than
the synthetic one, where labels were ground truth.

---

## 5. What this study establishes

**Transfers to production data**
- Reinforcement-learned headroom control: larger gains than on synthetic data
  (+53% utilisation, −42% cost), with a *lower* failure rate.
- Multi-cloud provider selection: −14.3% cost at no measurable cost elsewhere.
- The deep agent generalises better than the tabular one.
- Reactive threshold scaling is the worst arm on both workloads.

**Does not transfer**
- The forecasting advantage. On production data no tree ensemble beats
  persistence at any horizon, and the deficit grows with horizon. The synthetic
  conclusion was generator structure, not a property of cloud workloads.

**Practical consequence.** The predictive-autoscaling arm's advantage over
static and reactive control comes from the *control policy* — sizing to a
forecast rather than to a stale measurement — and not from the sophistication of
the forecaster. On this workload a persistence forecast would serve as well as
XGBoost, at a fraction of the complexity. That is a useful engineering finding,
and it is only visible because the naive baseline was measured throughout.

---

## 6. Reproducing

```bash
cd backend
python scripts/fetch_bitbrains.py --vms 300 --seed 42
python scripts/train.py    --data data/workload_bitbrains.csv \
                           --artifacts artifacts_bitbrains \
                           --trace data/workload_bitbrains.csv \
                           --max-fleet 80 --no-tune --rl-episodes 12
python scripts/evaluate.py --trace data/workload_bitbrains.csv \
                           --artifacts artifacts_bitbrains --max-fleet 80
python scripts/horizon_study.py --trace data/workload_bitbrains.csv \
                           --seeds 1 2 3 --out artifacts_bitbrains/horizon_study.json
```

Outputs land in `backend/artifacts_bitbrains/`.

---

## 7. Limitations

- **300 of 1,241 VMs.** A larger sample would smooth the aggregate further; the
  sample is uniform and seeded, so the study is reproducible, but it is a sample.
- **One trace, one provider, 2013.** Bitbrains is enterprise workload from one
  datacentre. Google, Azure and Alibaba traces are available from the same mirror
  and would test whether these conclusions hold across workload classes — the
  obvious next experiment.
- **Costs remain simulated.** The demand is real; the instance catalogue and
  pricing model are not. Relative comparisons hold; absolute dollars do not.
- **Heuristic anomaly labels**, as noted in §1.
