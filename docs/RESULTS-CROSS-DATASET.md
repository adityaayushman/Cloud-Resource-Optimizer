# Five workloads, one question: does any of this generalise?

`docs/RESULTS.md` evaluates the system on a workload generator written for this
project. `docs/RESULTS-REAL-TRACE.md` repeats it on one real trace and finds that
half the conclusions reverse. Neither answers the question a reader should ask
next: **was that reversal a property of cloud workloads, or a property of that
one trace?**

This study runs the whole evaluation on five workloads — the synthetic generator
and four public production traces — and separates the claims that replicate from
the claims that do not.

The short version:

- **Reinforcement-learned headroom control replicates on every workload it was
  run against** — all four of them. Cost falls 19–42% against a predictive
  baseline each time. (Alibaba is excluded from the control experiment for a
  documented reason; see §4.)
- **Multi-cloud provider selection replicates on all four**, at −13% to −15%,
  with every other metric unchanged.
- **The forecasting conclusion does not generalise, and neither does its
  reversal.** Whether a learned forecaster beats a persistence baseline depends
  on the workload — and it is predictable in advance from one cheap statistic,
  before any model is trained.

That last point is the contribution, and it is stronger than a qualitative claim:
order the five workloads by that statistic and you have also ordered them by how
well a learned forecaster does against persistence, with **no inversions**
(r = +0.956).

Bitbrains — the trace the earlier study drew its conclusion from — turns out to
be the *only* one of the five where any model loses to persistence, and the only
one whose demand behaves like a random walk. Generalising from it would have been
exactly as wrong as generalising from the synthetic generator, which is what the
study before it did.

---

## 1. The five workloads

All four real traces come from the same mirror, in a consistent per-entity CSV
layout, and are rebuilt by one script:

```bash
python scripts/fetch_trace.py --dataset {bitbrains|google|azure|alibaba}
```

| | Source | Entities | Rows | Span | Mean CPU | RAM:CPU |
|---|---|---|---|---|---|---|
| **synthetic** | generator in `app/workload.py` | — | 8,640 | 30 d | 14.7 | 2.40 |
| **bitbrains** | GWA-T-12 enterprise VMs, 2013 | 300 / 1,241 | 8,640 | 30 d | 136.7 | 1.26 |
| **google** | Borg cluster tasks, 2019 | 300 / 1,635 | 8,929 | 31 d | 40.2 | 3.13 |
| **azure** | Azure VM CPU readings | 300 / 1,195 | 8,639 | 30 d | 47.9 | 2.00\* |
| **alibaba** | production cluster machines, 2018 | 400 / 1,000 | 1,769 | 6.1 d | 14,451 | 9.12 |

\* Azure publishes **no memory column**. Its RAM is derived at a fixed 2 GB/core
and flagged `ram_is_synthetic` in the manifest. Azure therefore appears in the
CPU-only forecasting study on equal terms, and its ablation should be read as a
CPU-bound simulation — with RAM at exactly the ratio of the `large` instance
type, memory never binds.

Every entity sample is uniform, seeded, and recorded in
`data/workload_*_manifest.json` alongside the unit conversions.

### Two ingestion problems worth recording

**Alibaba's `_percent` columns are fractions.** `cpu_util_percent` and
`mem_util_percent` in this mirror hold values in [0, 1], not [0, 100]. Verified
across 40 randomly sampled machine files: no value exceeds 1.0 in either column,
and mean CPU is 0.395 — i.e. ~40% utilisation, which is what Alibaba reports for
this cluster. Reading them as percentages understates demand by 100×. The
adapter now detects the convention rather than assuming it.

**Aggregate demand needs a coverage filter, and the filter needs to be gentle.**
Entities do not all span the same wall-clock window, so a naive sum ramps up and
down at the edges purely as an artefact of how many entities were reporting.
Alibaba genuinely needs this — 473 edge slots trimmed. But a first version that
treated *every* low-coverage slot as a hard break was far worse than the disease:
Bitbrains coverage is a healthy 279–294 entities throughout, yet ~25 scattered
one-slot dips fragmented it so badly that the longest gap-free run was 2,972 of
8,640 slots. Short dips are now interpolated and only a sustained outage splits
the series.

That repair mattered more than its size suggests. The 25 fake dips were
suppressing Bitbrains's first-difference autocorrelation from **+0.173 to
+0.078** — an artificial down-and-back-up spike is exactly a negative
first-difference correlation. The corrected figure makes Bitbrains a *more*
clear-cut random walk, which strengthens rather than weakens the conclusion
below.

### What the traces actually look like

| trace | CV | acf1 | **diff_acf1** | mean step |
|---|---|---|---|---|
| synthetic | 0.683 | 0.976 | −0.349 | 9.76% |
| **bitbrains** | 0.573 | 0.984 | **+0.173** | 4.88% |
| google | 0.183 | 0.841 | −0.521 | 8.06% |
| azure | 0.105 | 0.947 | −0.319 | 2.73% |
| alibaba | 0.243 | 0.895 | −0.222 | 8.31% |

`acf1` — the autocorrelation of the *level* — is above 0.84 everywhere and says
almost nothing: all cloud demand is smooth at five-minute resolution. The
informative statistic is **`diff_acf1`, the lag-1 autocorrelation of the first
difference**:

- **Near zero** means a random walk. This interval's change carries no
  information about the next one, so "next equals current" is already the optimal
  forecast and no model can systematically beat it.
- **Strongly negative** means changes reverse — demand overshoots and settles.
  Persistence cannot exploit that structure; a model can.

Bitbrains is the only trace on the wrong side of that line, and it is the trace
the earlier single-dataset study drew its conclusion from.

---

## 2. Protocol

Both halves of the study are more conservative than the single-dataset versions
they replace.

**Rolling-origin walk-forward evaluation.** Rather than one 70/15/15 split, each
model is refit at K successive origins, trained on all history before each test
block. Test blocks are **disjoint**, so the K paired margins approximate
independent observations rather than K views of one test set.

Training-set size grows across blocks, deliberately. Restricting it to a short
fixed window handicaps only the learned arms, because persistence needs no
training at all — on synthetic demand XGBoost loses to persistence with a 7-day
window and ties it with three weeks. `--sliding` reproduces the fixed-window
variant as a robustness check.

**Paired significance testing.** Per (dataset, horizon, model), the K per-block
differences go through a two-sided **Wilcoxon signed-rank** test against zero —
signed-rank rather than a *t*-test because K is small and per-block error is not
normally distributed.

**Multiple-comparison correction.** 5 datasets × 4 horizons × 3 models = 60
hypotheses; at α = 0.05 three would clear by chance alone. All p-values are
adjusted by **Holm–Bonferroni** across the whole family, and both raw and
adjusted values are reported.

**MAE ratio as the primary measure.** R² is reported for continuity with the rest
of the project, but it is computed against the variance of whichever block it
lands in, which moves with the block. The MAE ratio (model ÷ persistence; < 1
means the model wins) is scale-free and block-independent, and the conclusions
rest on it.

**No hyperparameter tuning, for any arm.** Library defaults throughout, so the
ranking cannot be an artefact of unequal search budgets.

Reproduce with:

```bash
python scripts/cross_dataset_study.py                  # forecasting, all 5
./scripts/run_trace_study.sh google 40                 # train + ablate one trace
```

---

## 3. Forecasting: it depends on the workload, and the dependence is predictable

### Verdict by workload

MAE ratio of the *best* model at each horizon (model ÷ persistence; below 1 means the model wins), with the Holm-adjusted verdict.

| trace | diff_acf1 | 5 min | 15 min | 30 min | 60 min |
|---|---|---|---|---|---|
| **synthetic** | -0.349 | ➖ 0.95 | ➖ 0.95 | ➖ 0.95 | ➖ 0.64 |
| **bitbrains** | +0.173 | ❌ 1.25 | ❌ 1.12 | ❌ 1.09 | ➖ 1.14 |
| **google** | -0.521 | ✅ 0.84 | ✅ 0.84 | ✅ 0.87 | ✅ 0.87 |
| **azure** | -0.319 | ✅ 0.88 | ✅ 0.89 | ✅ 0.87 | ➖ 0.90 |
| **alibaba** | -0.222 | ➖ 0.96 | ➖ 0.89 | ➖ 0.80 | ➖ 0.83 |

✅ a learned model beats persistence, significant after Holm correction · ❌ persistence beats every model, significant · ➖ no significant difference

Read the ticks carefully. After Holm correction across all 60 tests, most cells
are **ties** — the effects here are real but modest, and a protocol that reports
every nominal p < 0.05 as a finding would be overstating them. What survives
correction is the shape of the table, not any single cell.

**Bitbrains is the only workload where a model ever loses significantly**, and it
loses in 11 of its 12 cells. No other trace contains a single significant loss.

### Does `diff_acf1` predict the outcome?

| trace | diff_acf1 | mean MAE ratio | cells won | cells lost | cells tied |
|---|---|---|---|---|---|
| **synthetic** | -0.349 | 0.999 | 1 | 0 | 11 |
| **bitbrains** | +0.173 | 1.564 | 0 | 11 | 1 |
| **google** | -0.521 | 0.958 | 4 | 0 | 8 |
| **azure** | -0.319 | 1.012 | 3 | 0 | 9 |
| **alibaba** | -0.222 | 1.053 | 0 | 0 | 12 |

Correlation between a trace's `diff_acf1` and its mean MAE ratio across all 12 of its cells: **+0.956** (n = 5 workloads).

That correlation is the result. Order the five workloads by `diff_acf1` and you
have also ordered them by how well a learned forecaster does against persistence
— bitbrains, alibaba, azure, synthetic, google, worst to best, with no
inversions. The statistic costs one pass over a trace and no training.

Three further observations.

**Linear regression is the only model that ever wins on real data.** Of the seven
significant wins across the three real traces, all seven are linear regression
(Google 4, Azure 3). XGBoost and Random Forest never once beat persistence on a
production trace at any horizon — and on Bitbrains they lose *worse* than linear
regression does (MAE ratio 1.84 vs 1.44 at 60 minutes). Mean reversion is linear
structure; trees split on local thresholds and extrapolate noise. The synthetic
generator is the sole workload where an ensemble wins, and it wins there because
the generator contains learnable step structure — the batch window and the
weekday-afternoon amplification — that a linear model cannot represent.

**The synthetic breakeven is 60 minutes, not 15.** The claim in `docs/RESULTS.md`
came from three overlapping windows with no significance test. Under disjoint
blocks and Holm correction, only Random Forest at 60 minutes survives
(p = 0.017); XGBoost and linear regression at the same horizon land at p = 0.060
and 0.075 and are reported as ties.

**Alibaba wins nothing and loses nothing.** All twelve cells tie. It has the
fewest test blocks (16, from 6.1 days), so it has the least statistical power of
the five — an honest "we could not tell" rather than a null result.

### Per-workload detail

#### synthetic — Generator written for this project

*23 disjoint test blocks · diff_acf1 = -0.349*

| Horizon | Model | MAE ratio | Blocks won | p (raw) | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| 5 min | XGBoost | 1.160 | 13/23 | 0.731 | 1.000 | ties |
| 5 min | Random Forest | 1.106 | 12/23 | 0.482 | 1.000 | ties |
| 5 min | Linear Regression | 0.994 | 14/23 | 0.259 | 1.000 | ties |
| 15 min | XGBoost | 1.161 | 9/23 | 0.080 | 1.000 | ties |
| 15 min | Random Forest | 1.125 | 10/23 | 0.119 | 1.000 | ties |
| 15 min | Linear Regression | 0.988 | 14/23 | 0.315 | 1.000 | ties |
| 30 min | XGBoost | 1.065 | 13/23 | 0.964 | 1.000 | ties |
| 30 min | Random Forest | 1.058 | 13/23 | 0.941 | 1.000 | ties |
| 30 min | Linear Regression | 0.969 | 15/23 | 0.119 | 1.000 | ties |
| 60 min | XGBoost | 0.748 | 19/23 | 0.001 | 0.060 | ties |
| 60 min | Random Forest | 0.729 | 19/23 | 0.000 | 0.017 | **beats** |
| 60 min | Linear Regression | 0.889 | 19/23 | 0.002 | 0.075 | ties |

#### bitbrains — Bitbrains GWA-T-12, enterprise VMs, 2013

*23 disjoint test blocks · diff_acf1 = +0.173*

| Horizon | Model | MAE ratio | Blocks won | p (raw) | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| 5 min | XGBoost | 1.532 | 0/23 | 2e−7 | 1e−5 | loses |
| 5 min | Random Forest | 1.428 | 0/23 | 2e−7 | 1e−5 | loses |
| 5 min | Linear Regression | 1.431 | 1/23 | 5e−6 | 0.000 | loses |
| 15 min | XGBoost | 1.710 | 1/23 | 2e−6 | 9e−5 | loses |
| 15 min | Random Forest | 1.533 | 1/23 | 2e−6 | 9e−5 | loses |
| 15 min | Linear Regression | 1.260 | 1/23 | 5e−6 | 0.000 | loses |
| 30 min | XGBoost | 1.833 | 1/23 | 2e−6 | 0.000 | loses |
| 30 min | Random Forest | 1.641 | 1/23 | 3e−6 | 0.000 | loses |
| 30 min | Linear Regression | 1.313 | 5/23 | 0.000 | 0.006 | loses |
| 60 min | XGBoost | 1.838 | 5/23 | 0.000 | 0.013 | loses |
| 60 min | Random Forest | 1.817 | 5/23 | 0.000 | 0.013 | loses |
| 60 min | Linear Regression | 1.435 | 7/23 | 0.006 | 0.235 | ties |

#### google — Google Borg cluster tasks, 2019

*23 disjoint test blocks · diff_acf1 = -0.521*

| Horizon | Model | MAE ratio | Blocks won | p (raw) | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| 5 min | XGBoost | 1.017 | 17/23 | 0.119 | 1.000 | ties |
| 5 min | Random Forest | 0.945 | 19/23 | 0.008 | 0.297 | ties |
| 5 min | Linear Regression | 0.854 | 21/23 | 0.000 | 0.008 | **beats** |
| 15 min | XGBoost | 1.006 | 17/23 | 0.105 | 1.000 | ties |
| 15 min | Random Forest | 0.957 | 17/23 | 0.020 | 0.685 | ties |
| 15 min | Linear Regression | 0.852 | 21/23 | 3e−5 | 0.002 | **beats** |
| 30 min | XGBoost | 1.053 | 12/23 | 0.393 | 1.000 | ties |
| 30 min | Random Forest | 1.003 | 14/23 | 0.731 | 1.000 | ties |
| 30 min | Linear Regression | 0.887 | 20/23 | 1e−5 | 0.001 | **beats** |
| 60 min | XGBoost | 1.026 | 12/23 | 0.917 | 1.000 | ties |
| 60 min | Random Forest | 1.000 | 16/23 | 0.482 | 1.000 | ties |
| 60 min | Linear Regression | 0.900 | 19/23 | 0.000 | 0.015 | **beats** |

#### azure — Microsoft Azure VM CPU readings

*23 disjoint test blocks · diff_acf1 = -0.319*

| Horizon | Model | MAE ratio | Blocks won | p (raw) | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| 5 min | XGBoost | 1.073 | 14/23 | 0.870 | 1.000 | ties |
| 5 min | Random Forest | 1.019 | 16/23 | 0.482 | 1.000 | ties |
| 5 min | Linear Regression | 0.884 | 23/23 | 2e−7 | 1e−5 | **beats** |
| 15 min | XGBoost | 1.106 | 15/23 | 0.823 | 1.000 | ties |
| 15 min | Random Forest | 1.045 | 15/23 | 0.754 | 1.000 | ties |
| 15 min | Linear Regression | 0.885 | 22/23 | 5e−7 | 3e−5 | **beats** |
| 30 min | XGBoost | 1.105 | 15/23 | 0.846 | 1.000 | ties |
| 30 min | Random Forest | 1.055 | 15/23 | 0.520 | 1.000 | ties |
| 30 min | Linear Regression | 0.880 | 22/23 | 1e−5 | 0.001 | **beats** |
| 60 min | XGBoost | 1.079 | 15/23 | 0.988 | 1.000 | ties |
| 60 min | Random Forest | 1.082 | 13/23 | 0.893 | 1.000 | ties |
| 60 min | Linear Regression | 0.935 | 17/23 | 0.007 | 0.283 | ties |

#### alibaba — Alibaba production cluster machines, 2018

*16 disjoint test blocks · diff_acf1 = -0.222*

| Horizon | Model | MAE ratio | Blocks won | p (raw) | p (Holm) | Verdict |
|---|---|---|---|---|---|---|
| 5 min | XGBoost | 1.159 | 4/16 | 0.008 | 0.283 | ties |
| 5 min | Random Forest | 1.046 | 9/16 | 0.433 | 1.000 | ties |
| 5 min | Linear Regression | 0.972 | 14/16 | 0.044 | 1.000 | ties |
| 15 min | XGBoost | 1.096 | 8/16 | 0.782 | 1.000 | ties |
| 15 min | Random Forest | 1.088 | 10/16 | 0.980 | 1.000 | ties |
| 15 min | Linear Regression | 0.984 | 12/16 | 0.348 | 1.000 | ties |
| 30 min | XGBoost | 1.067 | 11/16 | 0.597 | 1.000 | ties |
| 30 min | Random Forest | 1.154 | 11/16 | 0.669 | 1.000 | ties |
| 30 min | Linear Regression | 1.006 | 11/16 | 0.348 | 1.000 | ties |
| 60 min | XGBoost | 0.978 | 11/16 | 0.323 | 1.000 | ties |
| 60 min | Random Forest | 1.051 | 10/16 | 0.464 | 1.000 | ties |
| 60 min | Linear Regression | 1.036 | 12/16 | 0.597 | 1.000 | ties |

---

## 4. Control: the reinforcement-learning result replicates everywhere

Full system versus the ML-only predictive baseline, 7 policies × 3 seeds × 288
ticks, identical trace window per seed:

| trace | base util | full util | ΔUtil | ΔCost | base fail | full fail | nodes base→full |
|---|---|---|---|---|---|---|---|
| synthetic | 60.3% | 77.6% | **+28.7%** | **−33.6%** | 0.24% | 1.37% | 9.2 → 6.1 |
| bitbrains | 58.2% | 89.1% | **+53.0%** | **−42.2%** | 0.08% | 0.95% | 19.2 → 10.9 |
| google | 71.8% | 76.2% | **+6.0%** | **−19.5%** | 0.26% | 0.66% | 15.5 → 11.5 |
| azure | 76.2% | 87.9% | **+15.4%** | **−26.8%** | 0.00% | 0.22% | 7.0 → 5.7 |

Four workloads, four wins on both utilisation and cost, and the trade-off is the
same one every time: failures rise from near-zero to well under 1.4%. This is a
different point on the cost/reliability frontier, not a free lunch — but it is a
*consistent* point, which the single-dataset study could not establish.

**The size of the gain tracks how much the baseline was already wasting.** The
predictive baseline holds a fixed headroom multiplier that has to be sized for
the peaks; the learned policy adapts headroom to state. So the recoverable slack
is whatever the fixed multiplier left on the table, and across these four
workloads the gain correlates with the baseline's own utilisation at
**r = −0.84** — lowest baseline, largest gain:

| trace | baseline util | ΔUtil | CV |
|---|---|---|---|
| bitbrains | 58.2% | **+53.0%** | 0.573 |
| synthetic | 60.3% | +28.7% | 0.683 |
| google | 71.8% | +6.0% | 0.183 |
| azure | 76.2% | +15.4% | 0.105 |

Burstiness is the plausible underlying cause — a fixed multiplier has to be sized
for the peaks, so a burstier workload wastes more — but it is the *weaker*
predictor here (r = +0.74 against ΔUtil) and it does not order the four
correctly: Azure is the smoothest workload yet gains more than Google. With four
points neither correlation should be taken as established; the mechanism is
clear, the coefficient is not evidence.

Alibaba is excluded from the ablation, not from the forecasting study. Its
RAM:CPU ratio is 9.1 while the richest instance type in the catalogue offers 4.0,
so every policy would be permanently memory-bound and the ablation would be
measuring catalogue mismatch rather than control policy. That is a real finding
about the catalogue, and it is reported rather than worked around.

### Multi-cloud selection: the most reliably replicated result in the project

| trace | baseline $/day | with provider selection | saving |
|---|---|---|---|
| synthetic | 15.10 | 13.17 | −12.8% |
| bitbrains | 119.99 | 102.84 | −14.3% |
| google | 42.48 | 36.10 | −15.0% |
| azure | 38.16 | 32.31 | −15.3% |

Response latency is **identical** in all four pairs, and on Google and Azure so
are utilisation and failure rate, to the last digit recorded. On the other two the
movement is within noise: utilisation shifts by 0.33 points on synthetic and 0.16
on Bitbrains, failure rate by 0.04 and 0.001 points, against arm standard
deviations an order of magnitude larger. Provider selection changes where capacity
is bought and essentially nothing else — four independent confirmations of a −13%
to −15% saving at no measurable cost.

### The reactive arm collapses on Google

On Google, the threshold-reactive arm does not merely underperform, it fails
outright:

| trace | `diff_acf1` | reactive fail % | reactive latency | reactive SLA |
|---|---|---|---|---|
| **google** | −0.521 | **67.55%** | **59,765 s** | **3.1%** |
| synthetic | −0.349 | 1.07% | 1,720 s | — |
| azure | −0.319 | 2.88% | 5,195 s | 92.6% |
| bitbrains | +0.173 | 12.30% | 6,675 s | 80.8% |

Two thirds of all work rejected, and a mean recovery latency of sixteen hours.
This is the strongest argument in the whole project against reactive threshold
autoscaling, and it is worth stating plainly: on one of four production workloads
the industry-standard control policy did not merely lose to the learned one, it
was unusable.

The tempting explanation is that Google is also the most strongly mean-reverting
trace, and that a reactive controller — which scales to the *last* observation,
exactly as a persistence forecaster does — is always correcting toward a level
that has already gone away. The mechanism is real, but **these four points do not
establish it**: the correlation between `diff_acf1` and the reactive failure rate
is only −0.465, and synthetic is the second-most mean-reverting workload while
having the *lowest* reactive failure rate of the four. Google is a striking single
case with a plausible mechanism, not a demonstrated relationship, and separating
the two would need more workloads than this study has.

### Deep versus tabular, a fourth time

| trace | Q-learning fail % | DQN fail % |
|---|---|---|
| synthetic | 5.25% | 1.62% |
| bitbrains | 1.78% | 1.10% |
| google | 11.26% | 1.66% |
| azure | 0.65% | 0.44% |

The tabular agent reaches comparable or higher utilisation every time and pays
for it in rejected work, badly so on Google. Four replications of the same
generalisation gap.

---

## 5. What this study establishes

**Replicates across all workloads tested**

- Reinforcement-learned headroom control: −19% to −42% cost against a predictive
  baseline, at a failure rate under 1.4%.
- Multi-cloud provider selection: −13% to −15% cost at no measurable cost
  elsewhere.
- The deep agent generalises better than the tabular one.

**Does not replicate — and should never have been stated as a general claim**

- Any fixed verdict on learned forecasting. The synthetic study concluded the
  forecaster wins from 15 minutes out; the Bitbrains study concluded it never
  wins. Both were true of their own dataset and neither generalises.

**The finding that replaces it.** `diff_acf1` is computable in one line of numpy
from a trace you already have, before any model is trained, and it separates the
workloads where a learned forecaster earns its complexity from the workloads
where a two-line persistence baseline is the correct engineering answer. Across
these five it does so without a single inversion (r = +0.956), and it ships as
`GET /api/workload/forecastability`.

**A finding about model choice, which was not the question asked.** Every one of
the seven significant wins on production data is *linear regression*. The tree
ensembles — the models this project built its pipeline around, and the ones the
synthetic study ranked first — never beat persistence on a real trace at any
horizon, and on Bitbrains they lose by more than the linear model does. If this
system were rebuilt today, XGBoost would not be the default predictor.

---

## 6. Limitations

- **Four traces, one per provider, spanning 2013–2019.** Better than one, still
  not a sample of "cloud workloads".
- **Alibaba is 6.1 days**, against 30 for the others, and contributes fewer test
  blocks and correspondingly less statistical power.
- **Azure memory is derived, not measured**, and is used only where CPU-only
  claims are made.
- **Entity sampling.** 300–400 of 1,000–1,635 available per trace. Uniform and
  seeded, so reproducible, but a sample: a larger draw would smooth each
  aggregate further and could shift `CV` and `diff_acf1` somewhat.
- **Costs remain simulated.** Demand is real; the instance catalogue and pricing
  model are not. Relative comparisons hold, absolute dollars do not.
- **`diff_acf1` is a diagnostic supported by five points, not a law.** The
  headline r = +0.956 is a correlation over **five** workloads, and a correlation
  on five points is easy to obtain by chance — with n = 5 it is not even
  significant at α = 0.05 on its own. What carries the argument is not the
  coefficient but the mechanism: a random walk has no exploitable structure in
  its increments, which is a mathematical fact rather than an empirical one, and
  Bitbrains is the only workload here on that side of the boundary. Five points
  are enough to show the dependence exists and to falsify the single-dataset
  claims; they are not enough to fit a threshold anyone should trust as a
  constant, which is why the diagnostic reports "inconclusive" over a wide band
  rather than guessing.
- **Most cells tie.** After Holm correction, 41 of the 60 cells show no
  significant difference; 11 are losses (all Bitbrains) and 8 are wins. The study
  is better at ruling claims out than at establishing them, and it is reported
  that way.
