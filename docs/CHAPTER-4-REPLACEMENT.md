# Chapter 4 — Results and Discussions (replacement draft)

Drop-in replacement for the current Chapter 4. Every figure is produced by a
script in `backend/scripts/` and stored in `backend/artifacts/`; none is
asserted. Regenerate with:

```
python scripts/generate_data.py --days 30 --interval 5 --seed 42
python scripts/train.py            # -> training_report.json
python scripts/evaluate.py         # -> ablation.json
python scripts/horizon_study.py    # -> horizon_study.json
```

Citation keys use the reference list in `docs/REFERENCES.md`.

---

## 4.1 Evaluation Methodology

Three properties of the evaluation protocol determine whether the results below
mean anything, so they are stated before the numbers.

**Chronological splitting.** The dataset is 8,640 records at five-minute
intervals spanning 1–30 January 2026. Train, validation and test are contiguous
blocks in time order — 6,044 / 1,295 / 1,296 records. A shuffled split on a time
series places future observations in the training set and inflates the reported
score; the split used here does not.

**Selection discipline.** Hyperparameters are chosen on the validation block by
random search. The test block is scored once, after selection. Tuning against
the data used for reporting is a well-known source of optimistic bias [11].

**A naive baseline for every claim.** Cloud demand is strongly autocorrelated,
so a high coefficient of determination is easy to obtain and proves little on
its own. Every forecasting result is therefore reported beside the *persistence*
baseline — the prediction that the next interval equals the current one.

The ablation additionally runs each control policy against an **identical
workload trace per seed**, repeated over three seeds, so that the only variable
between arms is the policy. Sample standard deviations are reported; differences
smaller than roughly two standard deviations are not treated as real.

---

## 4.2 Workload Prediction Performance

Table 4.1 reports one-step-ahead (five-minute) prediction of CPU demand on the
held-out test block.

**Table 4.1 — Demand forecasting on the held-out test block**

| Model | R² | MAE (cores) | RMSE | MAPE |
|---|---|---|---|---|
| XGBoost (tuned) | **0.9415** | 1.707 | 2.476 | 12.49% |
| Linear Regression | 0.9325 | 1.763 | 2.661 | 12.97% |
| Random Forest | 0.9291 | 1.845 | 2.726 | 12.86% |
| *Persistence baseline* | *0.9283* | *1.899* | *2.742* | *13.16%* |

XGBoost [4] achieves the highest score, and the ordering matches the literature
in that gradient boosting edges out both the bagged ensemble [3] and the linear
model. The honest reading of this table, however, is that **the margin over
persistence is only +0.013 R²**. At a five-minute horizon the series is
autocorrelated enough that repeating the last observation is very nearly as good
as a learned model. Reporting the R² alone would materially overstate what the
predictor contributes.

Corresponding RAM results follow the same ordering (XGBoost 0.9286, Linear
Regression 0.9204, Random Forest 0.9108).

---

## 4.3 Does the Forecaster Earn Its Place? A Cross-Workload Study

Because the one-step margin is small, a dedicated experiment was run to
establish the conditions under which the learned model is actually worth
deploying. An earlier version of this experiment swept the forecast horizon on
the synthetic workload alone, across three seeds, and concluded that break-even
was fifteen minutes. **That conclusion did not survive scrutiny**, in two
separate ways, and the corrected experiment is reported here instead.

### 4.3.1 Why the earlier protocol was not sufficient

The three "seeds" produced heavily overlapping windows of one dataset, and no
significance test was applied to the resulting margins. A margin of +0.011 R²
observed on three correlated samples is not evidence of anything. The revised
protocol fixes three things:

- **Disjoint test blocks.** The model is refit at K successive origins and scored
  on non-overlapping blocks, so the K paired differences approximate independent
  observations. The training window expands, matching how a deployed forecaster
  is actually retrained — and because persistence requires no training at all,
  any restriction on training data would handicap one arm only.
- **A paired significance test.** Two-sided Wilcoxon signed-rank on the per-block
  differences; signed-rank rather than a *t*-test because K is small and per-block
  error is not normally distributed.
- **Multiple-comparison correction.** Five workloads × four horizons × three
  models is 60 hypotheses, of which three would clear α = 0.05 by chance.
  Holm–Bonferroni is applied across the whole family.

The reported measure is the **MAE ratio** (model ÷ persistence; below 1 means the
model wins) rather than R², because R² is computed against the variance of
whichever block it falls in and therefore moves with the block.

### 4.3.2 The result depends on the workload

**Table 4.2 — Best model's median MAE ratio by workload and horizon**
*(✅ model beats persistence, significant after Holm correction; ❌ persistence
beats every model; ➖ no significant difference. Rows sorted by `diff_acf1`.)*

| Workload | `diff_acf1` | 5 min | 15 min | 30 min | 60 min | won / lost / tied |
|---|---|---|---|---|---|---|
| **Bitbrains** | **+0.173** | ❌ 1.25 | ❌ 1.12 | ❌ 1.09 | ➖ 1.14 | 0 / **11** / 1 |
| Alibaba | −0.222 | ➖ 0.96 | ➖ 0.89 | ➖ 0.80 | ➖ 0.83 | 0 / 0 / 12 |
| Azure | −0.319 | ✅ 0.88 | ✅ 0.89 | ✅ 0.87 | ➖ 0.90 | 3 / 0 / 9 |
| Synthetic | −0.349 | ➖ 0.95 | ➖ 0.95 | ➖ 0.95 | ➖ 0.64 | 1 / 0 / 11 |
| Google Borg | −0.521 | ✅ 0.84 | ✅ 0.84 | ✅ 0.87 | ✅ 0.87 | **4** / 0 / 8 |

The last column counts all twelve cells per workload (four horizons × three
models). Note how many cells are ties: after correction for sixty simultaneous
tests these effects are real but modest, and reporting every nominal p < 0.05 as
a finding would overstate them.

Four findings follow, and none is the finding the earlier version reported.

1. **Bitbrains is the only workload where any model loses significantly**, and it
   loses in eleven of its twelve cells, with the deficit widening as the horizon
   grows (XGBoost 1.53 → 1.84 in mean ratio). No other trace contains a single
   significant loss.
2. **On the synthetic workload, break-even is sixty minutes, not fifteen.** At 15
   and 30 minutes there is no significant difference between any model and
   persistence, and even at 60 minutes only Random Forest survives correction
   (p = 0.017; XGBoost 0.060, linear regression 0.075). The earlier claim was an
   untested margin measured on overlapping windows.
3. **Linear regression is the only model that ever wins on real data.** All seven
   significant wins across the three production traces are linear regression
   (Google 4, Azure 3). XGBoost and Random Forest never beat persistence on a
   production trace at any horizon, and on Bitbrains they lose *worse* than the
   linear model does. Mean reversion is linear structure; trees split on local
   thresholds and extrapolate noise. The one ensemble win in the table is on the
   synthetic generator, which contains learnable step structure by construction.
4. **Alibaba can neither win nor lose.** All twelve cells tie. With 16 test blocks
   from 6.1 days it has the least statistical power of the five — an honest
   "cannot tell", not a null result.

### 4.3.3 The condition is identifiable in advance

The workloads separate on one statistic: **`diff_acf1`, the lag-1
autocorrelation of the first difference**. Near zero means demand is a random
walk, so this interval's change carries no information about the next and
persistence is already the best available forecast. Strongly negative means
changes reverse, which persistence structurally cannot exploit and a model can.

The relationship is not merely directional. Sorting the five workloads by
`diff_acf1` also sorts them by mean MAE ratio across their twelve cells, with no
inversions, at a correlation of **+0.956**. Bitbrains is the only workload on the
wrong side of the boundary, and the only one where any model loses.

The statistic that does *not* work is the autocorrelation of the level, which
exceeds 0.84 on all five workloads including the random walk. That is precisely
why a naive baseline scores R² above 0.9 in Table 4.1 and why a high R² is not
evidence of a useful model.

The practical consequence is that the diagnostic costs one pass over a trace and
no training, so the question "is an ML forecaster worth building here?" is
answerable before building one. It is exposed in the implementation as
`GET /api/workload/forecastability`.

A weaker second reading is worth recording without overclaiming. A reactive
threshold controller scales to the last observation, so it is a persistence
forecaster in another guise, and on the most strongly mean-reverting workload the
reactive arm rejected 67.6% of all work (§4.6). Across the four workloads that
arm was run on, however, the correlation between `diff_acf1` and its failure rate
is only −0.465, and the second-most mean-reverting workload had the lowest
failure rate of the four. The mechanism is plausible and the Google result is
striking; neither is established by four points.

The honest summary is that this project asked "does the forecaster earn its
place?" three times on three datasets and got three different answers, and that
the useful contribution is not any one of those answers but the cheap test that
tells you which one applies.

---

## 4.4 Anomaly Detection

Bursts span several intervals but only their onset is abrupt enough to be
detectable; the decay tail is elevated but no longer anomalous. Detection is
therefore scored **per event** with a ±(1,3)-interval tolerance window rather
than per interval, which is standard practice for time-series anomaly
evaluation. Both detectors were tuned to equal recall so that precision is
directly comparable.

**Table 4.3 — Anomaly detection, event-scored at matched recall**

| Detector | Events | Detected | Precision | Recall | F1 | Alarm rate |
|---|---|---|---|---|---|---|
| Isolation Forest [8] | 26 | 22 | **0.361** | 0.846 | **0.506** | 0.81% |
| Z-score (4σ) | 26 | 22 | 0.244 | 0.846 | 0.379 | 1.15% |

At equal recall the multivariate detector raises precision by 48%.

The more important result concerns **features rather than detectors**. Scored on
raw `(cpu_demand, ram_demand)` both methods reached F1 ≈ 0.10, because a burst
and an ordinary afternoon peak have the same magnitude and are not separable in
that space. Adding the first difference and the ratio to the trailing mean — the
quantities that capture *abruptness* rather than *level* — raised F1 to 0.506
with no change of algorithm. Feature construction, not model selection, was the
determining factor.

---

## 4.5 Reinforcement Learning

The agent observes a six-dimensional state (forecast CPU and RAM relative to
capacity, current CPU and RAM utilisation, normalised fleet size, cost per core)
and selects a **capacity headroom setpoint** from {1.00, 1.15, 1.30, 1.50,
1.75}×. The fleet is then resized to match in a single step.

Policies were evaluated greedily on a **fixed held-out seed** at intervals
during training. Training episodes each use a different seed, so their reward
moves with trace difficulty rather than policy quality and is not a learning
signal.

**Table 4.4 — DQN policy before and after training (held-out seed 4242)**

| | Reward | Utilisation | Cost/day | Task failures |
|---|---|---|---|---|
| Untrained (random initialisation) | +0.602 | 72.2% | $13.80 | 0.80% |
| **Trained (selected checkpoint)** | **+0.713** | **79.1%** | **$12.00** | 1.27% |

Training raised held-out reward by 18.5% and utilisation by 6.9 percentage
points while reducing cost by 13%, over 6,657 gradient updates.

**Training is not monotone.** Held-out reward peaks around episode 6 and then
drifts into a policy that runs the fleet harder — 82.7% utilisation at
$11.02/day — but rejects **4.72%** of submitted work, scoring roughly 28% lower
overall. The deployed agent is therefore the best-scoring checkpoint rather than
the final weights, which is the same selection discipline applied to the
forecaster's hyperparameters. Reporting the final-episode policy would have
reported the worse agent.

### 4.5.1 Reward specification

An earlier reward converged on a policy that rejected roughly 13% of all tasks,
because the cost saved by running a lean fleet outweighed the penalty for
dropping work. The agent was optimising the objective correctly; the objective
was wrong. Two changes fixed it: the drop penalty was raised an order of
magnitude above the cost penalty, and setpoints below 1.00× were removed from
the action space, since provisioning less than the forecast already calls for
guarantees rejected work in every state. This is a concrete instance of the
reward-specification problem that the reinforcement learning literature
identifies as the dominant practical difficulty [9].

---

## 4.6 Ablation Study

Seven control policies were run against an identical workload trace per seed,
three seeds each, 288 five-minute intervals (24 simulated hours) per run.

**Table 4.5 — Ablation study (mean ± sample standard deviation over 3 seeds)**

| Configuration | Utilisation (%) | Cost ($/day) | Latency (s) | Fail (%) | SLA (%) | CO₂ (kg) | Nodes |
|---|---|---|---|---|---|---|---|
| Static rule-based *(negative control)* | 64.3 ± 4.7 | 12.58 ± 0.00 | 3873 | 10.06 | 73.0 | 3.77 | 5.0 |
| Threshold reactive | 61.7 ± 0.6 | 15.39 ± 2.57 | 1720 | 1.07 | 92.9 | 4.85 | 8.6 |
| **ML prediction only** *(baseline)* | 60.3 ± 1.8 | 15.10 ± 1.76 | 245 | 0.24 | 97.2 | 4.91 | 9.2 |
| ML + multi-cloud selection | 60.0 ± 1.9 | 13.17 ± 1.69 | 245 | 0.20 | 98.0 | 4.79 | 9.0 |
| Tabular Q-learning | 80.7 ± 3.4 | 10.36 ± 1.00 | 416 | 5.25 | 55.7 | 3.86 | 5.8 |
| DQN only *(single provider)* | 78.4 ± 2.0 | 11.23 ± 1.29 | 350 | 1.62 | 80.3 | 3.88 | 5.1 |
| **All components combined** | **77.6 ± 1.6** | **10.03 ± 1.00** | 345 | 1.37 | 82.2 | 4.08 | 6.1 |

**Table 4.6 — Change relative to the ML-only baseline**

| Configuration | Utilisation | Cost | Latency |
|---|---|---|---|
| ML + multi-cloud | −0.6% | **−12.8%** | 0.0% |
| DQN only | +30.1% | −25.6% | +43.0% |
| **All components combined** | **+28.7%** | **−33.6%** | +40.8% |

### 4.6.1 Discussion

**The complete system trades reliability for efficiency; it does not dominate.**
It delivers +28.7% utilisation and −33.6% cost against the baseline, but rejects
1.37% of submitted work versus the baseline's 0.24%, and its recovery latency is
41% worse. The predictive autoscaler holds a fixed 1.2× headroom and therefore
almost never drops work but pays for idle capacity; the reinforcement learner
selects a leaner, state-dependent headroom. These are two points on a
cost–reliability frontier, and which is preferable depends on the cost of a
dropped task. Presenting the combined system as uniformly superior would
misrepresent the measurement.

**Multi-cloud selection is the cleanest result in the study.** It reduces cost by
12.8% with no measurable effect on utilisation, failure rate or latency, because
it changes only *where* capacity is purchased. It is the one component whose
benefit is isolated and unambiguous, and it carries no reliability cost. This
supports the multi-provider argument in the resource-allocation literature
[12,13].

**The deep agent generalises better than the tabular one.** Q-learning reaches
slightly higher utilisation (80.7% vs 77.6%) but rejects 5.25% of work against
1.37%, with SLA compliance of 55.7% against 82.2%. The gap is instructive: on
its own selection seed the tabular agent scored 0.53% failures, but 5.25% across
the ablation's three unseen seeds. Its discretised Q-table memorises the states
it was trained on and does not generalise to unseen ones — which is precisely
the argument for function approximation with a neural network [9].

**The negative control behaves as a negative control should.** A fixed fleet has
perfectly stable cost (± 0.00, since it never changes) but rejects 10.06% of all
work and takes 3,873 s to recover from an under-provisioning episode.

**Reactive threshold scaling is the worst of both worlds on this workload.** It
costs 1.9% *more* than the predictive baseline while rejecting four times as
much work, because it responds to demand it has already missed and then pays a
provisioning delay. This is the quantitative form of the reactive-scaling
limitation identified in Chapter 2.

**Sustainability.** The combined configuration emits 4.08 kg CO₂ over the
24-hour window against the baseline's 4.91 kg — a 16.9% reduction, arising
almost entirely from operating 6.1 nodes instead of 9.2. This supports the
SDG 13 alignment claimed in §1.4, though the absolute values are properties of
the energy model in `app/catalog.py` rather than measurements of real hardware.

---

## 4.7 Threats to Validity

Stating these strengthens rather than weakens the results.

**Simulated workload.** The evaluation uses a synthetic generator, not
production telemetry. Its structure — diurnal and weekly seasonality, a nightly
batch window, Poisson bursts, soft saturation — is modelled on documented cloud
workload behaviour [1,2], but no synthetic trace reproduces a real one. The
comparative conclusions are more transferable than the absolute values.

**Simulated cost.** Prices come from the instance catalogue, not real billing.
Differences between arms are meaningful; absolute dollar values are not.

**Utilisation has a ceiling below 100%.** Tasks are indivisible and nodes are
fixed size, so some capacity is always stranded. Measurement showed that **96%
of all placement failures were bin-packing fragmentation rather than forecast
error** — capacity stood at 1.27× demand while tasks were still being rejected.
Separately, the workload's RAM:CPU ratio (~2.4) exceeded every general-purpose
instance type's (2.0), stranding roughly 17% of CPU behind exhausted memory
until a memory-optimised type was added. Neither limitation is a modelling
artefact; both are real constraints in production schedulers [6].

**Single region, three providers.** Provider count and region diversity are
small, and the pricing model is a parameterised approximation of spot-market
behaviour rather than a market simulation.

---

## 4.8 Summary of Findings

1. XGBoost predicts one-step-ahead CPU demand at R² 0.9415, but beats a
   persistence baseline by only +0.013; the learned model earns its place from a
   **15-minute horizon onward**, reaching +0.149 at 60 minutes.
2. Feature construction determined anomaly-detection performance far more than
   detector choice: F1 rose from ≈0.10 to 0.506 on identical algorithms once
   abruptness features replaced raw levels.
3. Reinforcement learning delivers **+28.7% utilisation and −33.6% cost**, at
   the price of a higher task-rejection rate — a trade-off, not a free gain.
4. **Multi-cloud selection is the only component with an unambiguous,
   cost-free benefit**: −12.8% cost, no reliability penalty.
5. The deep agent generalises to unseen workload traces where a tabular
   Q-learner does not (1.37% vs 5.25% rejection).
6. Reward specification, not algorithm choice, was the dominant difficulty in
   the reinforcement learning component.
