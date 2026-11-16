"""Figures for README / MODEL.md. Re-run after every model change:

    python -m docs.gen_model_figs

Outputs:
    docs/model_metrics.png        — test ρ and MSE per development phase
    docs/uplift_distribution.png  — anatomy of the uplift target
    docs/current_metrics.png      — headline metrics of the current model
"""
from __future__ import annotations

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.graph.visualize import BLUE, GREEN, GREY, ORANGE, RED  # shared style (rcParams)

DOCS = Path(__file__).parent
ROOT = DOCS.parent
GRAPH = ROOT / "data" / "graph"
MODELS = GRAPH / "models"
SEASON = "2025-2026"

# Manual record — one row per phase, updated by hand after each training run.
# (label, test ρ, test MSE, comparable?)  — first two use the leaky setup.
PHASES = [
    ("Baseline: joint target, random split", 0.789, 0.155, False),
    ("1.1 player-disjoint split",            0.736, 0.190, False),
    ("1.2 uplift target",                    0.334, 0.166, True),
    ("1.3 + cold-start eval",                0.325, 0.172, True),
    ("1.4 + team/league embeddings",         0.312, 0.173, True),
    ("1.5 + ranking loss λ=1, weighted",     0.334, 0.211, True),
    ("1.6 data fix: missing teams",          0.368, 0.169, True),
    ("1.7 ranking loss λ=0.5",               0.371, 0.167, True),
    ("1.8 position fix (labels only)",       0.371, 0.167, True),
]


def _save(fig, name: str) -> None:
    out = DOCS / name
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {out}")


def fig_metrics() -> None:
    labels = [p[0] for p in PHASES][::-1]
    rho = np.array([p[1] for p in PHASES])[::-1]
    mse = np.array([p[2] for p in PHASES])[::-1]
    honest = np.array([p[3] for p in PHASES])[::-1]
    y = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True,
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    c = [BLUE if h else "#cbd5e1" for h in honest]
    c[0] = GREEN  # latest phase
    ax1.barh(y, rho, color=c)
    ax2.barh(y, mse, color=[RED if h else "#cbd5e1" for h in honest])
    for yi, v in zip(y, rho):
        ax1.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9)
    for yi, v in zip(y, mse):
        ax2.text(v + 0.003, yi, f"{v:.3f}", va="center", fontsize=9)
    ax1.set_yticks(y, labels, fontsize=9.5)
    ax1.set_xlim(0, 0.9); ax2.set_xlim(0, 0.25)
    ax1.set_xlabel("test Spearman ρ  (higher is better)")
    ax2.set_xlabel("test MSE  (lower is better)")
    ax1.set_title("Ranking quality", loc="left")
    ax2.set_title("Prediction error", loc="left")
    ax1.axhline(len(labels) - 2.5, color=GREY, lw=0.8, ls="--")
    ax1.text(0.88, len(labels) - 2.35, "grey: leaky setup, not comparable",
             ha="right", fontsize=8.5, color=GREY)
    fig.suptitle(f"SynergyHGT development history · season {SEASON}", fontweight="bold",
                 fontsize=14, x=0.02, ha="left")
    fig.tight_layout()
    _save(fig, "model_metrics.png")


def fig_uplift_distribution() -> None:
    f = GRAPH / f"edges_teammate_{SEASON}.parquet"
    if not f.exists():
        print(f"skipped (missing): {f}")
        return
    df = pd.read_parquet(f)
    if "per90_uplift" not in df.columns:
        print(f"skipped (no per90_uplift): {f}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1.hist(df["per90_uplift"], bins=90, range=(-1.5, 3.0), color=GREEN, alpha=0.9)
    ax1.axvline(0, color="#0f172a", lw=1, ls="--", label="0 = no synergy")
    ax1.axvline(df["per90_uplift"].mean(), color=RED, lw=1.5,
                label=f"mean = {df['per90_uplift'].mean():.3f}")
    ax1.set_xlabel("per90_uplift"); ax1.set_ylabel("teammate pairs")
    ax1.set_title("Target distribution", loc="left")
    ax1.legend()

    lim = float(np.percentile(np.r_[df["expected_per90"], df["per90_joint_output"]], 99.5))
    hb = ax2.hexbin(df["expected_per90"], df["per90_joint_output"], gridsize=50,
                    extent=(0, lim, 0, lim), mincnt=1, cmap="Blues", bins="log", linewidths=0)
    ax2.plot([0, lim], [0, lim], ls="--", color=RED, lw=1, label="observed = expected")
    ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)
    ax2.set_xlabel("expected per-90 output (solo a + solo b)")
    ax2.set_ylabel("observed per-90 output together")
    ax2.set_title("Uplift = distance above the diagonal", loc="left")
    ax2.legend(loc="upper left")
    fig.colorbar(hb, ax=ax2, label="pairs (log)", shrink=0.85)
    fig.tight_layout()
    _save(fig, "uplift_distribution.png")


def fig_current_metrics_panel() -> None:
    f = MODELS / f"metrics_{SEASON}.json"
    if not f.exists():
        print(f"skipped (missing): {f}")
        return
    m = json.loads(f.read_text())
    cold = m.get("cold_start") or {}
    info = m.get("split_info") or {}
    pl, eg = info.get("players", {}), info.get("edges", {})

    cards = [
        ("Test Spearman ρ", f"{m['test_spearman']:.3f}", "pairs with ≥1 unseen player", BLUE),
        ("Cold-start ρ", f"{cold.get('spearman', float('nan')):.3f}",
         f"both players unseen · n={cold.get('n_edges', '?'):,}", ORANGE),
        ("Test MSE", f"{m['test_mse']:.3f}", "per-90 uplift units", RED),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 2.9))
    for ax, (title, value, sub, color) in zip(axes, cards):
        ax.axis("off")
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                   fc="#f8fafc", ec="#e2e8f0", lw=1.2))
        ax.add_patch(plt.Rectangle((0, 0), 0.025, 1, transform=ax.transAxes, fc=color, ec="none"))
        ax.text(0.08, 0.78, title, fontsize=12, color="#334155", transform=ax.transAxes)
        ax.text(0.08, 0.38, value, fontsize=30, weight="bold", color="#0f172a",
                transform=ax.transAxes)
        ax.text(0.08, 0.13, sub, fontsize=9.5, color=GREY, transform=ax.transAxes)
    fig.suptitle(
        f"Current model · {SEASON} · split: {m.get('split')} · players train/val/test "
        f"{pl.get('train', '?'):,} / {pl.get('val', '?'):,} / {pl.get('test', '?'):,} · "
        f"edges {eg.get('train', '?'):,} / {eg.get('val', '?'):,} / {eg.get('test', '?'):,}",
        fontsize=10, color="#475569", x=0.01, ha="left", y=1.02)
    fig.tight_layout()
    _save(fig, "current_metrics.png")


if __name__ == "__main__":
    fig_metrics()
    fig_uplift_distribution()
    fig_current_metrics_panel()
