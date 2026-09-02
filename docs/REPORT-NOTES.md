# Notes for the written report

This file maps the implementation onto the project report and the conference
paper, and flags the places where the two must be brought into agreement before
submission. It exists because several claims in the current report are not
supported by any code that produces them.

---

## 1. Numbers that must be replaced

The report's Table 4.1 and Table 3.4 give a specific set of figures. None of them
were produced by a measurement harness. This repository now produces measured
equivalents in `artifacts/ablation.json` via `scripts/evaluate.py`, and
`docs/RESULTS.md` carries the current table.

Replace, don't reconcile — the two sets are not the same experiment.

| Report claim | Status |
|---|---|
| Utilisation 62.4% → 89.7% (+43.7%) | Replace with measured values |
| Task failure 4.2% → 0.5% (−88.1%) | Replace |
| Scaling latency 180 s → 15 s (−91.7%) | Replace; latency is now defined as mean under-provisioned episode duration plus a 45 s modelled boot time, and the units are not comparable |
| Cost $145.20 → $102.35/day (−29.5%) | Replace; absolute cost depends on fleet scale and pricing model |
| R² 0.94 / MAE 6.8% (XGBoost) | Measured R² is 0.9415 — close, but state it as one-step-ahead and give the split protocol |
| LR R² 0.78 | Measured LR R² is 0.9325. The report's 0.78 is not reproducible with a causal lag feature present |

Also note: the report's MAE column is expressed as a percentage. That is MAPE,
not MAE. `docs/RESULTS.md` reports both under their correct names.

## 2. Claims the code now supports that it previously did not

These were marked "Complete" in the report's sprint tables but had no working
implementation. They now do, and can be demonstrated live:

- **US-09 / US-16 (explainability).** Exact TreeSHAP per prediction, rendered as
  a diverging bar chart on the Prediction page. Endpoint:
  `GET /api/session/{id}/explain`.
- **US-14 / US-15 (anomaly detection and alerts).** Isolation Forest and z-score
  detectors, both wired into the advisory panel and the engine log.
- **US-18 (multi-cloud distribution).** Nodes are genuinely provisioned across
  AWS, Azure and GCP; the topology panel shows the split. Previously every node
  was created on AWS because `add_vm` defaulted to it and no call site overrode it.
- **US-11 (RL policy updates from rewards).** The DQN's action changes the fleet,
  and the reward is measured from the resulting interval.

## 3. Structural corrections needed in the report

Independent of the code:

1. **Table 2.1 caption** says "Summary of Related Work in **RUL Prediction**" —
   wrong domain, left over from another document.
2. **Table 4.2 caption** says "(lower RMSE is better)" but the table has no RMSE
   column.
3. **§3.2 title** differs between the table of contents ("Regime-Aware Adaptation
   and Ablation Study") and the body. "Regime-Aware Adaptation" appears nowhere
   else.
4. **Appendix A** contains a `load_and_process_data()` function that parses
   `id`/`cycle`/`s1..s21` columns and computes Remaining Useful Life from a
   turbofan dataset. It has nothing to do with cloud workloads.
5. **No reference is cited in the body.** The bibliography lists [1]–[17] and none
   appear in the text. Breiman, Chen & Guestrin, Mnih and Lundberg & Lee — the
   four papers behind the methods actually used — are missing entirely. The
   conference paper cites them correctly; port that list across.
6. **Appendix C (plagiarism report)** is an empty page.
7. **Fig 4.1 and Fig 4.3** have non-monotonic y-axes (120, 150, 110, 110 and 380,
   250, 220, 210, 200). They need regenerating.
8. **Fig 3.2** shows training/validation loss over 20 epochs for a model that
   Table 3.1 describes as an XGBoost regressor with 100 trees. Tree ensembles do
   not have epochs, and the code has no validation split producing that curve.
   The DQN learning curve on the Results page is a genuine substitute.
9. **Fig 2.1** shows "Flask Web App" and "REST API Endpoints". The application is
   FastAPI + React; regenerate the diagram from the architecture table in the
   README.
10. **User-story IDs collide** between §2.5 (backlog) and §3.1.1/§3.2.1 (sprints):
    US-06 means "scale proactively" in one and "implement RL allocation" in the
    other. Renumber one scheme.
11. **§3.2.2** says "we added three new features" and then lists four.
12. **§2.6** says the project was "broken down into two phases" and then describes
    three.

## 4. Methodology the report should now describe

Points worth stating explicitly, because they are what makes the results
defensible under questioning:

**Split protocol.** Train/validation/test are contiguous blocks in time order
(70/15/15). Hyperparameters are chosen on validation; the test block is scored
once. A shuffled `train_test_split` on a time series leaks future observations
into training, and tuning against the test set inflates the reported score — the
original design did both.

**Causal features.** Lag and rolling statistics are computed with an explicit
`.shift(1)` before the rolling window, so the feature row for time *t* never
contains the target at *t*. Warm-up rows are dropped rather than back-filled;
back-filling a lag feature copies a future value backwards.

**Forecast horizon.** The target is demand at *t+1* given features at *t*. This is
the quantity an autoscaler needs — capacity must exist before demand arrives.

**Persistence baseline.** Every forecast result is reported beside "next interval
equals this one". Without it, a high R² on an autocorrelated series says nothing.

**Event-based anomaly scoring.** A burst spans several intervals but only its
onset is detectable; the decay tail is elevated but not abrupt. Detection is
therefore scored per event with a tolerance window, and the two detectors are
compared at matched recall so precision is meaningful.

**Controlled ablation.** All seven policies run on an identical workload trace per
seed, repeated over three seeds, with the standard deviation reported. Only the
control policy varies.

## 5. Honest findings worth reporting

These emerged from measurement and are more interesting than the original claims:

1. **Reinforcement learning is not free.** With a naive reward the DQN converged
   on a lean fleet that rejected 13% of tasks — it was correctly optimising a
   badly-specified objective. Raising the drop penalty an order of magnitude above
   the cost penalty, and removing sub-1.0 headroom setpoints from the action
   space, was necessary before RL beat the predictive baseline on anything other
   than cost. This is a genuine result about reward design.
2. **96% of placement failures were bin-packing fragmentation**, not forecast
   error — capacity sat at 1.27× demand while tasks were still rejected. Task
   granularity relative to node size mattered more than model accuracy.
3. **Instance shape matters more than instance size.** The workload's RAM:CPU
   ratio (~2.4) exceeded every general-purpose node's (2.0), stranding ~17% of CPU
   behind exhausted memory until a memory-optimised type was added.
4. **Multi-cloud selection is the cleanest win.** It reduces cost with no effect
   on utilisation, failure rate or latency, because it changes only where nodes
   are bought — an isolated, unambiguous effect.
5. **Linear regression is a strong baseline on smooth workloads.** The tree
   ensembles only pull ahead once the workload contains non-smooth structure
   (a nightly batch window, a weekday-afternoon interaction). Worth stating, as
   it justifies the choice of model rather than assuming it.

## 6. Appendix A

The appendix should be regenerated from the current source. The code in the
submitted report does not run: `engine.py` imports a `models` module that the
appendix never lists, `predict()` passes a raw NumPy array to models fitted on a
named DataFrame, and `fillna(method='bfill')` is removed in pandas 3.

Suggested appendix order, matching the module map in the README:
`models.py`, `catalog.py`, `workload.py`, `ml/predictor.py`, `ml/dqn.py`,
`ml/anomaly.py`, `engine.py`, `simulation.py`, `main.py`.
