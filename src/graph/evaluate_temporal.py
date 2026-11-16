"""Out-of-season test: a model trained on season S predicts pair synergy in season S+1.

The player-disjoint split measures generalisation to unseen players *within* a season.
This is the harder, real-world question: do the synergies the model predicts from last
season's data show up in next season's matches?

    python -m src.graph.evaluate_temporal --train 2025-2026 --test 2026-2027

Pairs are split into
  - repeat pairs: already teammates (≥90 min together) in the training season
  - new pairs:    first time together (transfers, new combinations) — the scouting case
Baselines (no GNN):
  - player-mean uplift: mean training-season uplift of a's pairs + of b's pairs
  - persistence (repeat pairs only): the pair's own training-season uplift
Only players present in the training graph can be scored; coverage is reported.

Output: data/graph/models/temporal_<train>_to_<test>.json and docs/temporal_test.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from src.graph.edge_builder import build_edges
from src.graph.recommend import SynergyEngine

GRAPH_DIR = Path("data/graph")
MIN_MINUTES = (90, 270, 450)  # noise in the observed uplift falls with shared minutes


def _rho(a, b) -> float | None:
    return float(spearmanr(a, b).correlation) if len(a) >= 20 else None


def _key(a: pd.Series, b: pd.Series) -> pd.Series:
    """Order-independent pair key."""
    return np.where(a < b, a + "|" + b, b + "|" + a)


def evaluate(train_season: str, test_season: str, rebuild: bool = True) -> dict:
    eng = SynergyEngine.load(train_season)
    train = pd.read_parquet(GRAPH_DIR / f"edges_teammate_{train_season}.parquet")
    test_path = GRAPH_DIR / f"edges_teammate_{test_season}.parquet"
    test = build_edges(test_season)["teammate"] if rebuild or not test_path.exists() \
        else pd.read_parquet(test_path)

    n_all = len(test)
    known = test["player_a"].isin(eng.pid_to_idx) & test["player_b"].isin(eng.pid_to_idx)
    test = test[known].reset_index(drop=True)

    # GNN prediction: mean of both directions (the edge MLP is not exactly symmetric)
    a = torch.tensor([eng.pid_to_idx[p] for p in test["player_a"]])
    b = torch.tensor([eng.pid_to_idx[p] for p in test["player_b"]])
    with torch.no_grad():
        pred = (eng.model.predict_edge(eng._emb, a, b) + eng.model.predict_edge(eng._emb, b, a)) / 2
    test["pred"] = pred.cpu().numpy()

    # Baselines from the training season's observed uplift
    long = pd.concat([train[["player_a", "per90_uplift"]].rename(columns={"player_a": "p"}),
                      train[["player_b", "per90_uplift"]].rename(columns={"player_b": "p"})])
    player_mean = long.groupby("p")["per90_uplift"].mean()
    global_mean = float(train["per90_uplift"].mean())
    test["base_player_mean"] = (test["player_a"].map(player_mean).fillna(global_mean)
                                + test["player_b"].map(player_mean).fillna(global_mean))
    prev = pd.Series(train["per90_uplift"].values, index=_key(train["player_a"], train["player_b"]))
    test["prev_uplift"] = pd.Series(_key(test["player_a"], test["player_b"])).map(prev).values
    test["repeat"] = test["prev_uplift"].notna()

    rows = []
    for min_m in MIN_MINUTES:
        sub = test[test["minutes_together"] >= min_m]
        for group, g in (("all", sub), ("new pairs", sub[~sub["repeat"]]), ("repeat pairs", sub[sub["repeat"]])):
            y = g["per90_uplift"]
            rows.append({
                "min_minutes": min_m, "group": group, "n": int(len(g)),
                "gnn": _rho(g["pred"], y),
                "player_mean": _rho(g["base_player_mean"], y),
                "persistence": _rho(g["prev_uplift"], y) if group == "repeat pairs" else None,
            })

    out = {
        "train_season": train_season, "test_season": test_season,
        "test_pairs_total": int(n_all), "test_pairs_scored": int(len(test)),
        "coverage": round(len(test) / n_all, 3) if n_all else 0.0,
        "results": rows,
    }
    path = GRAPH_DIR / "models" / f"temporal_{train_season}_to_{test_season}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")
    return out


def fig_temporal(res: dict, path: Path = Path("docs/temporal_test.png")) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.graph.visualize import BLUE, GREY, ORANGE

    groups = ["new pairs", "repeat pairs"]
    series = [("SynergyHGT", "gnn", BLUE), ("player-mean baseline", "player_mean", GREY),
              ("last season's pair uplift", "persistence", ORANGE)]
    fig, axes = plt.subplots(1, len(groups), figsize=(12, 4.6), sharey=True)
    for ax, grp in zip(axes, groups):
        rs = [r for r in res["results"] if r["group"] == grp]
        x = np.arange(len(rs)); w = 0.26
        for k, (label, key, color) in enumerate(series):
            vals = [r[key] if r[key] is not None else np.nan for r in rs]
            if all(np.isnan(vals)):
                continue
            bars = ax.bar(x + (k - 1) * w, vals, w, color=color, label=label)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.006, f"{v:.2f}",
                            ha="center", fontsize=8)
        ax.set_xticks(x, [f"≥{r['min_minutes']} min\nn={r['n']:,}" for r in rs])
        ax.axhline(0, color="#94a3b8", lw=0.8)
        ax.set_title(grp, loc="left")
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel(f"Spearman ρ vs observed {res['test_season']} uplift")
    for ax in axes:  # persistence only exists for repeat pairs → legend per panel
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Out-of-season test · trained on {res['train_season']}, "
                 f"tested on {res['test_season']} pairs", fontweight="bold", fontsize=13,
                 x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="2025-2026")
    ap.add_argument("--test", default="2026-2027")
    ap.add_argument("--no-rebuild", action="store_true", help="reuse the test-season edge parquet")
    a = ap.parse_args()
    res = evaluate(a.train, a.test, rebuild=not a.no_rebuild)
    print(f"coverage {res['coverage']:.1%} of {res['test_pairs_total']} test pairs")
    for r in res["results"]:
        print(r)
    fig_temporal(res)


if __name__ == "__main__":
    main()
