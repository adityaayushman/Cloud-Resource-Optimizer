# Measured results

Every figure here is produced by a script in `backend/scripts/` and written to
`backend/artifacts/`. Nothing is asserted by hand. Regenerate with:

```bash
python scripts/generate_data.py --days 30 --interval 5 --seed 42
python scripts/train.py
python scripts/evaluate.py --ticks 288 --repeats 3
python scripts/horizon_study.py
```

**Dataset.** 8,640 records at 5-minute intervals spanning 2026-01-01 → 2026-01-30
(30 days), from the generator in `app/workload.py`. Diurnal + weekly seasonality,
a weekday 02:00–04:00 batch window, a weekday-afternoon amplification, a Sunday
maintenance dip, soft arrival saturation, and 26 Poisson burst events.

---

## 1. Demand forecasting

One-step-ahead (5 min) CPU demand. Train/validation/test are **contiguous blocks
in time order** (70/15/15); hyperparameters selected on validation, test scored
once.

| Model | R² | MAE | RMSE | MAPE |
|---|---|---|---|---|
| **XGBoost** (tuned) | **0.9415** | 1.707 | 2.476 | 12.49% |
| Linear Regression | 0.9325 | 1.763 | 2.661 | 12.97% |
| Random Forest | 0.9291 | 1.845 | 2.726 | 12.86% |
| *Persistence baseline* | *0.9283* | *1.899* | *—* | *—* |

> **Read this table with the next one.** A high R² on an autocorrelated series
> is not evidence of a useful model. Persistence — "next interval equals this
> one" — already scores 0.9283. XGBoost's margin over it here is **+0.013**.

## 2. Does the forecaster earn its place? (horizon study)

`scripts/horizon_study.py` — 3 seeds × 30 days, margin = model R² − persistence R².

| Horizon | XGBoost | Random Forest | Linear Regression |
|---|---|---|---|
| 5 min | −0.0009 *(1/3 seeds)* | −0.0143 *(0/3)* | +0.0026 *(2/3)* |
| **15 min** | **+0.0108 (3/3)** | +0.0042 (2/3) | +0.0106 (2/3) |
| **30 min** | **+0.0403 (3/3)** | +0.0245 (2/3) | +0.0186 (2/3) |
| **60 min** | **+0.1490 (3/3)** | +0.1413 (3/3) | +0.0401 (2/3) |

**Conclusion.** At one 5-minute step the series is autocorrelated enough that
persistence is competitive and the learned model adds essentially nothing. The
forecaster separates decisively once it has to see past the autocorrelation —
breakeven is **15 minutes**, and by 60 minutes XGBoost leads persistence by
0.149 R². The ensembles also only pull away from the linear baseline at long
horizons (0.898 vs 0.789 at 60 min), which is where the workload's non-smooth
structure — the batch window, the weekday-afternoon interaction — starts to
matter more than the recent level.

This is the honest justification for the ML layer, and it is a stronger claim
than a bare R² because it is stated against a baseline and measured across seeds.

## 3. Anomaly detection

Scored **per burst event** with a ±(1, 3)-interval tolerance window. A burst
spans several intervals but only its onset is detectable, so point-wise scoring
would penalise a correct detector for the decay tail. Both detectors are tuned
to the same recall so precision is directly comparable.

| Detector | Events | Detected | Precision | Recall | F1 | Alarm rate |
|---|---|---|---|---|---|---|
| **Isolation Forest** | 26 | 22 | **0.361** | 0.846 | **0.506** | 0.81% |
| Z-score (4σ) | 26 | 22 | 0.244 | 0.846 | 0.379 | 1.15% |

At matched recall the multivariate detector raises precision by 48%. Feature
choice mattered more than detector choice: on raw `(cpu, ram)` both scored F1
≈ 0.10, because a burst and a normal afternoon peak are the same magnitude.
Adding first-difference and ratio-to-trailing-mean features is what made the
event separable.

## 4. Reinforcement learning

Policies are evaluated greedily on one **fixed held-out seed** (4242) at
intervals during training. Training episodes each use a different seed, so their
reward tracks trace difficulty rather than policy quality and is not a learning
signal.

| | Reward | Utilisation | Cost/day | Task failures |
|---|---|---|---|---|
| Untrained (random init) | +0.602 | 72.2% | $13.80 | 0.80% |
| **Trained (selected checkpoint)** | **+0.713** | **79.1%** | **$12.00** | 1.27% |

**Checkpoint selection is load-bearing.** DQN training here is not monotone: the
held-out reward peaks around episode 6, then drifts into a policy that trades
reliability for cost (82.7% utilisation, $11.02/day, but **4.72%** task
failures) and scores ~28% lower. The deployed agent is the best-scoring
checkpoint, not the final weights — the same discipline applied to the
forecaster's hyperparameters. Tabular Q-learning gets the identical protocol.

## 5. Ablation study

`scripts/evaluate.py` — 7 policies × 3 seeds × 288 ticks (24 h simulated). Every
arm sees an **identical workload trace per seed**; only the control policy
varies. Values are means over seeds, ± sample standard deviation.

| Configuration | Utilisation | Cost $/day | Latency | Fail % | SLA % | CO₂ kg | Nodes |
|---|---|---|---|---|---|---|---|
| Static rule-based *(negative control)* | 64.3 ±4.7 | 12.58 ±0.00 | 3873 s | 10.06 | 73.0 | 3.77 | 5.0 |
| Threshold reactive | 61.7 ±0.6 | 15.39 ±2.57 | 1720 s | 1.07 | 92.9 | 4.85 | 8.6 |
| **ML prediction only** *(baseline)* | 60.3 ±1.8 | 15.10 ±1.76 | 245 s | 0.24 | 97.2 | 4.91 | 9.2 |
| ML + multi-cloud | 60.0 ±1.9 | 13.17 ±1.69 | 245 s | 0.20 | 98.0 | 4.79 | 9.0 |
| Tabular Q-learning | 80.7 ±3.4 | 10.36 ±1.00 | 416 s | 5.25 | 55.7 | 3.86 | 5.8 |
| DQN only *(single provider)* | 78.4 ±2.0 | 11.23 ±1.29 | 350 s | 1.62 | 80.3 | 3.88 | 5.1 |
| **All components combined** | **77.6 ±1.6** | **10.03 ±1.00** | 345 s | 1.37 | 82.2 | 4.08 | 6.1 |

### Change vs the ML-only baseline

| Configuration | Utilisation | Cost | Latency |
|---|---|---|---|
| ML + multi-cloud | −0.6% | **−12.8%** | 0.0% |
| DQN only | +30.1% | −25.6% | +43.0% |
| **All combined** | **+28.7%** | **−33.6%** | +40.8% |

### What the table actually says

1. **The full system trades reliability for efficiency — it does not dominate.**
   It delivers +28.7% utilisation and −33.6% cost, but its task failure rate is
   1.37% against the baseline's 0.24%, and its response latency is 41% worse.
   The predictive autoscaler's fixed 1.2× headroom over-provisions, which is why
   it never drops work; the RL agent learns a leaner, state-dependent headroom.
   These are two different points on a cost/reliability frontier, and which is
   preferable depends on what a dropped task costs.

2. **Multi-cloud selection is the cleanest win in the study.** It cuts cost
   12.8% with no measurable effect on utilisation, failure rate or latency,
   because it changes only *where* nodes are bought. It is the one component
   with an unambiguous, isolated benefit.

3. **The deep agent beats the tabular one where it matters.** Q-learning reaches
   slightly higher utilisation (80.7% vs 77.6%) but at **5.25%** task failures
   against 1.37%, and 55.7% SLA compliance against 82.2%. It also generalises
   worse: on its own selection seed it scored 0.53% failures, but 5.25% across
   the ablation's three unseen seeds — the discretised Q-table overfits the
   states it was trained on, which is precisely the argument for the network.

4. **The negative control behaves like a negative control.** A fixed fleet is
   cheap and stable in cost ( ±0.00, it never changes) but drops 10% of all work
   and takes 3,873 s to recover from an under-provisioning episode.

5. **Reactive threshold scaling is the worst of both worlds here** — it costs
   *more* than the predictive baseline (+1.9%) while dropping 4× more work,
   because it chases demand it has already missed and pays a provisioning delay.

---

## 6. Notes on interpreting these numbers

**Latency** is the mean duration of an under-provisioned episode — from the tick
capacity falls below demand until it recovers — plus a modelled 45 s instance
boot. It is measured in units of the 300 s control interval, so it is not
comparable to a figure derived from a different tick granularity.

**Cost** is simulated from the instance catalogue in `app/catalog.py`, not from
real cloud billing. Relative differences between arms are meaningful; the
absolute dollar values are a property of the model.

**Utilisation** cannot reach 100%: tasks are indivisible and nodes are fixed
size, so some capacity is always stranded. Measurement showed 96% of all
placement failures came from this fragmentation rather than forecast error, and
that the workload's RAM:CPU ratio (~2.4) exceeded every general-purpose node's
(2.0) until a memory-optimised type was added.

**Standard deviations** are over 3 seeds. Differences smaller than roughly two
standard deviations should not be read as real — for instance the ML-only and
ML + multi-cloud arms are indistinguishable on utilisation (60.3 ±1.8 vs
60.0 ±1.9) and only separate on cost.
