"""Generate the report's figures from the committed artifacts.

Every figure the submitted report currently carries is either wrong or drawn from
nothing: Fig 4.1 and 4.3 have non-monotonic y-axes (120, 150, 110, 110), Fig 3.2
shows a 20-epoch training/validation loss curve for what Table 3.1 describes as a
100-tree XGBoost regressor - tree ensembles have no epochs and no such curve
exists in the code - and Fig 2.1 shows a Flask application that was never built.

These are drawn from `artifacts/*.json`, so a figure cannot disagree with the
table beside it. Re-run after any change to the measurements.

    python scripts/make_figures.py                 # -> artifacts/figures/*.png

Palette and mark conventions follow the project's data-visualisation rules: a
fixed categorical order (never cycled), one measure per axis (never a second
y-scale), direct value labels on bars - which is also what discharges the
sub-3:1 contrast warning on the lighter hues - and recessive grid and axes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Categorical slots in canonical order. Assigned in order and never cycled: a
# fifth series would fold into "other" rather than reuse slot 1.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8880"
SURFACE, GRID = "#fcfcfb", "#e4e3de"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": INK_3,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "legend.fontsize": 8,
})


def _clean(ax, *, xgrid=False):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.grid(axis="x" if xgrid else "y")
    ax.grid(axis="y" if xgrid else "x", visible=False)


def _save(fig, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Fig 2.1 - system architecture
# ---------------------------------------------------------------------------

def fig_architecture(out: Path) -> None:
    """Replaces the Flask/REST diagram with what was actually built."""
    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.set_xlim(0, 102); ax.set_ylim(0, 86); ax.axis("off")

    def box(x, y, w, h, title, sub, colour):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.4",
            linewidth=1.2, edgecolor=colour, facecolor=colour + "14"))
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="top",
                fontsize=8.5, fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 8.0, sub, ha="center", va="top",
                fontsize=6.9, color=INK_2, linespacing=1.5)

    def arrow(x1, y1, x2, y2, label="", lx=None, ly=None, ha="center"):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
            linewidth=1.0, color=INK_3))
        if label:
            # Labels sit in the gap between boxes, never over one.
            ax.text(lx if lx is not None else (x1 + x2) / 2,
                    ly if ly is not None else (y1 + y2) / 2 + 1.8,
                    label, ha=ha, va="center", fontsize=6.6, color=INK_2)

    ROW1, ROW2, H = 38, 2, 20
    box(1, ROW1, 20, H, "Presentation", "React 18 + Vite\non Vercel", BLUE)
    box(31, ROW1, 21, H, "Application", "FastAPI + uvicorn\non Render", ORANGE)
    box(62, ROW1, 21, H, "Processing",
        "SmartAllocator\nAutoScaler\nAdvisoryEngine", AQUA)
    box(31, ROW2, 21, H, "Learning",
        "XGBoost predictor\nNumPy DQN\nIsolation Forest", YELLOW)
    box(62, ROW2, 21, H, "Simulation",
        "Closed-loop harness\n7-policy ablation", AQUA)
    box(87, 19, 13, H, "Catalogue", "AWS · Azure\nGCP pricing", ORANGE)

    mid1, mid2 = ROW1 + H / 2, ROW2 + H / 2
    arrow(21.5, mid1, 30.5, mid1, "HTTPS", ly=mid1 + 3.2)
    arrow(52.5, mid1, 61.5, mid1, "step", ly=mid1 + 3.2)
    arrow(41.5, ROW1 - 0.5, 41.5, ROW2 + H + 0.5, "forecast", lx=43, ly=30, ha="left")
    arrow(52.5, mid2, 61.5, mid2, "replay", ly=mid2 + 3.2)
    arrow(72.5, ROW2 + H + 0.5, 72.5, ROW1 - 0.5, "reward", lx=74, ly=30, ha="left")
    arrow(83.5, mid1 - 4, 86.5, 40.5, "buy", lx=84, ly=45, ha="left")

    ax.text(0, 85, "Figure 2.1  System architecture", ha="left", va="top",
            fontsize=10, fontweight="bold", color=INK)
    ax.text(0, 77,
            "Replaces the submitted diagram, which showed a Flask application\n"
            "and REST endpoints that were never built.",
            ha="left", va="top", fontsize=7, color=INK_2, linespacing=1.6)
    _save(fig, out, "fig_2_1_architecture")


# ---------------------------------------------------------------------------
# Fig 3.2 - DQN learning curve (the genuine substitute for the fake loss curve)
# ---------------------------------------------------------------------------

def fig_learning_curve(out: Path, report: dict) -> None:
    rl = report["rl"]
    curve = rl["eval_curve"]
    episodes = [c["episode"] for c in curve]
    reward = [c["reward"] for c in curve]
    fails = [c["task_failure_rate"] for c in curve]
    selected = rl.get("selected_episode")

    # Two panels rather than two y-scales: reward and failure rate have unrelated
    # units, and a dual axis would let the choice of scaling imply a relationship.
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(6.4, 4.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.18})

    top.plot(episodes, reward, color=BLUE, linewidth=2, marker="o",
             markersize=4.5, markerfacecolor=BLUE, markeredgecolor=SURFACE,
             markeredgewidth=1.2, label="Held-out reward")
    if selected is not None:
        y = next(c["reward"] for c in curve if c["episode"] == selected)
        top.scatter([selected], [y], s=150, facecolor="none", edgecolor=ORANGE,
                    linewidth=2, zorder=5)
        top.annotate(f"deployed checkpoint\n(episode {selected})",
                     (selected, y), textcoords="offset points", xytext=(10, -26),
                     fontsize=7.5, color=INK_2,
                     arrowprops=dict(arrowstyle="-", color=INK_3, linewidth=0.8))
    top.set_ylabel("Mean episode reward")
    top.set_title("Figure 3.2  DQN training is not monotone", loc="left")
    span = max(reward) - min(reward)
    top.set_ylim(min(reward) - span * 0.12, max(reward) + span * 0.18)
    _clean(top)

    bottom.plot(episodes, fails, color=ORANGE, linewidth=2, marker="o",
                markersize=4.5, markerfacecolor=ORANGE,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    bottom.set_ylabel("Task failures (%)")
    bottom.set_xlabel("Training episode")
    _clean(bottom)

    fig.text(0.5, -0.04,
             "Evaluated greedily on a held-out seed after each episode. Reward peaks "
             "early and then drifts into a\npolicy that rejects more work, so the "
             "deployed agent is the best-scoring checkpoint, not the final weights.",
             ha="center", fontsize=7.5, color=INK_2, linespacing=1.5)
    _save(fig, out, "fig_3_2_dqn_learning")


# ---------------------------------------------------------------------------
# Fig 4.1 - ablation
# ---------------------------------------------------------------------------

SHORT = {
    "static_rules": "Static rules",
    "threshold_reactive": "Threshold reactive",
    "ml_predictive": "ML prediction only",
    "multicloud_only": "ML + multi-cloud",
    "q_learning": "Tabular Q-learning",
    "rl_only": "DQN only",
    "full": "All components",
}


def fig_ablation(out: Path, ablation: dict) -> None:
    rows = ablation["rows"]
    labels = [SHORT.get(r["strategy"], r["strategy"]) for r in rows]
    util = [r["utilisation"] for r in rows]
    cost = [r["cost_per_day"] for r in rows]
    fail = [r["task_failure_rate"] for r in rows]
    y = np.arange(len(rows))
    highlight = [r["strategy"] == "full" for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3), sharey=True)
    for ax, values, colour, title, fmt in (
            (axes[0], util, BLUE, "Utilisation (%)", "{:.1f}"),
            (axes[1], cost, ORANGE, "Cost ($/day)", "{:.2f}"),
            (axes[2], fail, AQUA, "Task failures (%)", "{:.2f}")):
        bars = ax.barh(y, values, height=0.62,
                       color=[colour if h else colour + "66" for h in highlight],
                       edgecolor=SURFACE, linewidth=1.4)
        # Direct labels on every bar: they are what makes the lighter hues legible
        # against the surface, and they remove the need to read values off a grid.
        span = max(values) or 1
        for bar, value in zip(bars, values):
            ax.text(bar.get_width() + span * 0.03, bar.get_y() + bar.get_height() / 2,
                    fmt.format(value), va="center", fontsize=7, color=INK_2)
        ax.set_title(title, loc="left", fontsize=9)
        ax.set_xlim(0, span * 1.28)
        _clean(ax, xgrid=True)

    axes[0].set_yticks(y, labels, fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle("Figure 4.1  Controlled ablation — 7 policies × 3 seeds × 24 h",
                 x=0.02, y=1.06, ha="left", fontsize=10, fontweight="bold",
                 color=INK)
    fig.text(0.02, -0.06,
             "Identical workload trace per seed; only the control policy varies. "
             "The full system is highlighted.\nIt is a different point on the "
             "cost/reliability frontier, not a strict improvement.",
             ha="left", fontsize=7.5, color=INK_2, linespacing=1.5)
    _save(fig, out, "fig_4_1_ablation")


# ---------------------------------------------------------------------------
# Fig 4.3 - the forecastability relationship
# ---------------------------------------------------------------------------

def fig_forecastability(out: Path, study: dict, analysis: dict | None) -> None:
    traces = study["traces"]
    rows = study["rows"]
    points = []
    for name, t in traces.items():
        cells = [r["mae_ratio_mean"] for r in rows if r["dataset"] == name]
        if cells:
            points.append((t["diff_acf1"], float(np.mean(cells)), name))
    points.sort()

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    pad = (xs.max() - xs.min()) * 0.12
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    # A shaded band rather than a text annotation on the line: the label collided
    # with whichever points sat near ratio 1.0, which is exactly where the
    # interesting ones are.
    ax.axhspan(1.0, ys.max() + 0.12, color=ORANGE, alpha=0.05, zorder=0)
    ax.axhline(1.0, color=INK_3, linewidth=1.0, linestyle=(0, (4, 3)), zorder=1)
    ax.set_ylim(ys.min() - 0.06, ys.max() + 0.08)

    beats = ys < 1.0
    ax.scatter(xs[beats], ys[beats], s=64, color=BLUE, edgecolor=SURFACE,
               linewidth=1.4, zorder=3, label="A learned model wins")
    ax.scatter(xs[~beats], ys[~beats], s=64, color=ORANGE, edgecolor=SURFACE,
               linewidth=1.4, zorder=3, label="Persistence wins")

    if len(xs) > 2:
        slope, intercept = np.polyfit(xs, ys, 1)
        grid = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 50)
        ax.plot(grid, slope * grid + intercept, color=INK_3, linewidth=1.2,
                linestyle=(0, (5, 3)), zorder=2)

    # Every point is labelled while there are few enough to read; past that only
    # the extremes and the workloads persistence wins on, which are the ones the
    # argument turns on.
    if len(points) <= 8:
        notable = {n for _, _, n in points}
    else:
        notable = {points[0][2], points[-1][2]} | {n for _, r, n in points if r >= 1.0}
    for i, (x, y_, name) in enumerate(points):
        if name not in notable:
            continue
        # Alternate the offset so neighbouring labels do not stack on each other.
        dy = 7 if i % 2 == 0 else -12
        ax.annotate(name, (x, y_), textcoords="offset points", xytext=(7, dy),
                    fontsize=6.8, color=INK_2)

    ax.set_xlabel("diff_acf1  —  lag-1 autocorrelation of the first difference")
    ax.set_ylabel("Mean MAE ratio  (model ÷ persistence)")
    ax.set_title("Figure 4.3  Forecastability is predictable before training",
                 loc="left")
    ax.legend(loc="upper left")
    _clean(ax)
    ax.grid(axis="x")

    caption = ("Each point is one workload; lower is better for the learned model, and "
               "persistence wins inside the\nshaded region. Demand that behaves like a "
               "random walk (diff_acf1 near zero) cannot be forecast\nbetter than by "
               "carrying the last value forward, so no model beats the baseline there.")
    if analysis:
        best = next((lv for lv in analysis["levels"]
                     if lv["level"].startswith("collection")), None)
        if best:
            caption += (f"\nAt the collection level: r = {best['pearson_r']:+.3f}, "
                        f"exact permutation p = {best['pearson_permutation_p']:.3g} "
                        f"(n = {best['n']}).")
    fig.text(0.02, -0.06, caption, ha="left", fontsize=7.5, color=INK_2,
             linespacing=1.5)
    _save(fig, out, "fig_4_3_forecastability")


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ROOT / "artifacts")
    ap.add_argument("--study", type=Path, default=None,
                    help="cross-dataset or panel study JSON for Figure 4.3")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or (args.artifacts / "figures")
    print(f"Writing figures to {out}")

    def load(name: str):
        path = args.artifacts / name
        if not path.exists():
            print(f"  skipping {name}: not found")
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    fig_architecture(out)

    report = load("training_report.json")
    if report:
        fig_learning_curve(out, report)

    ablation = load("ablation.json")
    if ablation:
        fig_ablation(out, ablation)

    study_path = args.study or (args.artifacts / "panel_study.json")
    if not study_path.exists():
        study_path = args.artifacts / "cross_dataset_study.json"
    if study_path.exists():
        study = json.loads(study_path.read_text(encoding="utf-8"))
        analysis_path = args.artifacts / "panel_analysis.json"
        analysis = (json.loads(analysis_path.read_text(encoding="utf-8"))
                    if analysis_path.exists() else None)
        fig_forecastability(out, study, analysis)
    else:
        print("  skipping Figure 4.3: no study JSON found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
