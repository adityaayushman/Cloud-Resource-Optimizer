# Report corrections — exact replacements

`REPORT-NOTES.md` says *what* is wrong with the submitted report. This says what
to put instead, so each item is a paste rather than a rewrite. Line references
are to the submitted `final report.pdf`.

Three of the items in the notes are now closed and are listed at the bottom
rather than repeated here.

---

## 1. Captions and headings

### Table 2.1

> **Currently:** "Summary of Related Work in RUL Prediction"

Wrong domain — left over from a turbofan remaining-useful-life document. Replace
with:

> **Table 2.1 — Summary of related work in cloud resource optimisation**

### Table 4.2

> **Currently:** "(lower RMSE is better)"

The table has no RMSE column. The measure it actually shows is the margin over
the persistence baseline, where **higher** is better. Replace with:

> **Table 4.2 — Margin over the persistence baseline by forecast horizon**
> *(margin = model R² − baseline R²; higher is better. A negative margin means
> the learned model is worse than assuming the next interval equals this one.)*

### §3.2 heading

The table of contents says "Regime-Aware Adaptation and Ablation Study"; the body
says something else, and "Regime-Aware Adaptation" appears nowhere in the
document or the code. Use the body title in both places, or if a single title is
wanted:

> **3.2  Reinforcement-Learned Allocation and Controlled Ablation**

---

## 2. Counts that contradict their own lists

### §3.2.2 — "we added three new features" then lists four

The anomaly detector takes **five** features, three of which were the later
addition. Replace the sentence with:

> The detector was initially given raw CPU and RAM demand and scored an F1 of
> 0.10, because a burst is not distinguished by its level but by its abruptness.
> Three further features were added — the first difference of CPU demand
> (`cpu_delta`), and the ratios of CPU and RAM demand to their trailing rolling
> means (`cpu_ratio`, `ram_ratio`) — giving five in total and raising F1 to 0.506.

### §2.6 — "broken down into two phases" then describes three

Replace "two phases" with "three phases", or if the intent was two, merge the
last two descriptions. The three as described are: data generation and
forecasting; reinforcement-learned allocation; multi-cloud selection and
evaluation.

---

## 3. User-story identifier collisions

`US-06` means "scale proactively" in the §2.5 backlog and "implement RL
allocation" in §3.1.1 / §3.2.1. Several other IDs collide the same way.

Renumbering either scheme fixes it, but the least disruptive change is to give
the two schemes different prefixes, since they are different things — a backlog
of user-facing requirements versus a list of sprint tasks:

| Scheme | Prefix | Example |
|---|---|---|
| §2.5 product backlog (user stories) | `US-` *(unchanged)* | US-06 "scale proactively" |
| §3.1.1 / §3.2.1 sprint tasks | **`SP-`** | SP-06 "implement RL allocation" |

Every `US-` reference inside §3.1.1 and §3.2.1 becomes `SP-`; references in §2.5
and the traceability matrix stay as they are. This keeps the backlog IDs stable,
which matters because they are what the requirements section is written against.

---

## 4. Citations

**No reference is cited anywhere in the body.** The bibliography lists [1]–[17]
and not one appears in the text; the four papers behind the methods actually used
— Breiman on Random Forests, Chen & Guestrin on XGBoost, Mnih et al. on DQN,
Lundberg & Lee on SHAP — are missing from the list entirely.

`docs/REFERENCES.md` carries a corrected list where every entry is cited and
every method used has its source, together with a citation-placement table
saying which reference belongs at which point in the text. Use that list
wholesale; the conference paper already cites these correctly.

---

## 5. Claims the measurements no longer support

These are not formatting problems. Two statements in the report are now
contradicted by this project's own evidence and must change.

### "Break-even is fifteen minutes"

Under the strict protocol — disjoint test blocks, a Wilcoxon signed-rank test on
paired per-block differences, Holm–Bonferroni across the family — the synthetic
break-even is **sixty** minutes, and only Random Forest survives correction
there. At 15 and 30 minutes ahead there is no significant difference between any
model and the baseline.

Worse for the original claim, it does not generalise at all: on Bitbrains no
model beats persistence at any horizon, while on Google and Azure linear
regression beats it at almost every horizon. See §4.3 of
`CHAPTER-4-REPLACEMENT.md`.

### "XGBoost is the best predictor"

Across four production traces, **every** significant win over the persistence
baseline belongs to linear regression. Neither XGBoost nor Random Forest beat
persistence on a real trace at any horizon, and on Bitbrains both lose by more
than the linear model does.

The report may keep XGBoost as the *implemented* default — it is what the system
ships, and it does win on the synthetic generator — but the results chapter has
to state that the measurement does not support it. Presenting it as the validated
best choice would be contradicted by the project's own results table.

---

## 6. Already closed

| Item | Where |
|---|---|
| Figures 2.1, 3.2, 4.1, 4.3 regenerated | `backend/artifacts/figures/`, from `scripts/make_figures.py` |
| Appendix A regenerated from working source | `docs/APPENDIX-A.md`, from `scripts/make_appendix.py` |
| Chapter 4 replacement drafted | `docs/CHAPTER-4-REPLACEMENT.md` |
| Reference list corrected | `docs/REFERENCES.md` |

## 7. Still needs a human

**Appendix C (plagiarism report)** is an empty page. It requires a Turnitin or
similar run through the institution's account — it cannot be generated from this
repository.
